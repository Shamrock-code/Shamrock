// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/Timer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

    using Patch     = shamrock::patch::Patch;
    using PatchTree = shamrock::scheduler::PatchTree;

    /// A patch tree with its leaves as patches (shuffled ids) and the corresponding coord transform
    template<class Tvec>
    struct TestPatchTree {
        PatchTree ptree;
        std::vector<Patch> leaves;
        shamrock::patch::PatchCoordTransform<Tvec> transform;
        shammath::CoordRange<Tvec> domain;
    };

    /**
     * @brief Make a patch tree with `root_grid` roots, all split `base_level` times, then
     * `extra_splits` splits of random leaves (so some branches are deeper)
     */
    template<class Tvec>
    TestPatchTree<Tvec> make_test_tree(
        std::mt19937 &eng,
        std::array<u32, 3> root_grid,
        u32 base_level,
        u32 extra_splits,
        shammath::CoordRange<Tvec> domain) {

        // same layout as PatchScheduler::make_patch_base_grid
        u32 max_lin = std::max({root_grid[0], root_grid[1], root_grid[2]});
        u64 sz_root = PatchScheduler::max_axis_patch_coord_length / sham::roundup_pow2_clz(max_lin);

        PatchTree ptree;
        u32 tmp_id = 0;
        for (u32 x = 0; x < root_grid[0]; x++) {
            for (u32 y = 0; y < root_grid[1]; y++) {
                for (u32 z = 0; z < root_grid[2]; z++) {
                    shamrock::patch::PatchCoord<3> coord;
                    coord.coord_min = {sz_root * x, sz_root * y, sz_root * z};
                    coord.coord_max
                        = {sz_root * (x + 1) - 1, sz_root * (y + 1) - 1, sz_root * (z + 1) - 1};
                    ptree.insert_root_node(tmp_id++, coord);
                }
            }
        }

        shamrock::patch::PatchCoord<3> bounds;
        bounds.coord_min = {0, 0, 0};
        bounds.coord_max
            = {sz_root * root_grid[0] - 1, sz_root * root_grid[1] - 1, sz_root * root_grid[2] - 1};

        for (u32 lev = 0; lev < base_level; lev++) {
            std::vector<u64> to_split(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : to_split) {
                ptree.split_node(id);
            }
        }

        // split random leaves (possibly ones that were just created)
        std::vector<u64> leaves(ptree.leaf_key.begin(), ptree.leaf_key.end());
        std::sort(leaves.begin(), leaves.end());
        for (u32 i = 0; i < extra_splits; i++) {
            u64 pick = std::uniform_int_distribution<u64>(0, leaves.size() - 1)(eng);
            u64 id   = leaves[pick];
            ptree.split_node(id);
            leaves[pick] = leaves.back();
            leaves.pop_back();
            for (u64 c : ptree.tree[id].tree_node.childs_nid) {
                leaves.push_back(c);
            }
        }

        // shuffled patch ids, like the global list ordered by owner rank rather than by id
        std::vector<u64> leaf_nodes(ptree.leaf_key.begin(), ptree.leaf_key.end());
        std::sort(leaf_nodes.begin(), leaf_nodes.end());
        std::vector<u64> ids(leaf_nodes.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), eng);

        for (auto &[nid, node] : ptree.tree) {
            node.linked_patchid = u64_max;
        }

        std::vector<Patch> patches;
        for (u64 i = 0; i < leaf_nodes.size(); i++) {
            auto &node          = ptree.tree[leaf_nodes[i]];
            node.linked_patchid = ids[i];

            Patch p{};
            p.id_patch = ids[i];
            p.override_from_coord(node.patch_coord);
            patches.push_back(p);
        }
        std::shuffle(patches.begin(), patches.end(), eng);

        return {
            std::move(ptree),
            std::move(patches),
            shamrock::patch::PatchCoordTransform<Tvec>{bounds.get_patch_range(), domain},
            domain};
    }

    template<class Tvec>
    Tvec rand_in_box(std::mt19937 &eng, Tvec lo, Tvec hi) {
        using Tscal = shambase::VecComponent<Tvec>;
        std::uniform_real_distribution<Tscal> dx(lo.x(), hi.x());
        std::uniform_real_distribution<Tscal> dy(lo.y(), hi.y());
        std::uniform_real_distribution<Tscal> dz(lo.z(), hi.z());
        return Tvec{dx(eng), dy(eng), dz(eng)};
    }

    /// Positions mostly in `box`, some on its faces, some anywhere (including outside `domain`)
    template<class Tvec>
    std::vector<Tvec> gen_positions(
        std::mt19937 &eng,
        u32 n,
        shammath::CoordRange<Tvec> box,
        shammath::CoordRange<Tvec> domain) {
        using Tscal = shambase::VecComponent<Tvec>;

        Tvec ext = (domain.upper - domain.lower) * Tscal(0.1);
        std::vector<Tvec> ret;
        std::uniform_int_distribution<u32> kind(0, 9);
        std::uniform_int_distribution<u32> axis(0, 2);
        for (u32 i = 0; i < n; i++) {
            u32 k = kind(eng);
            if (k < 6) {
                ret.push_back(rand_in_box(eng, box.lower, box.upper));
            } else if (k < 8) {
                // exactly on a lower (in) or upper (out) face of the box
                Tvec r = rand_in_box(eng, box.lower, box.upper);
                u32 a  = axis(eng);
                r[a]   = (k == 6) ? box.lower[a] : box.upper[a];
                ret.push_back(r);
            } else {
                ret.push_back(rand_in_box(eng, domain.lower - ext, domain.upper + ext));
            }
        }
        return ret;
    }

    template<class Tvec>
    bool vec_bit_equal(Tvec a, Tvec b) {
        return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
    }

    template<class T>
    std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len) {
        std::vector<T> ret(len);
        sycl::host_accessor acc{buf, sycl::read_only};
        for (u32 i = 0; i < len; i++) {
            ret[i] = acc[i];
        }
        return ret;
    }

    template<class Tvec>
    void test_compute_patch_owner(
        std::array<u32, 3> root_grid, u32 base_level, u32 extra_splits, u32 seed) {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        std::mt19937 eng(seed);

        shammath::CoordRange<Tvec> domain{Tvec{-1.1, -0.7, 0.3}, Tvec{2.3, 1.9, 1.7}};

        auto [ptree, leaves, transform, dom]
            = make_test_tree<Tvec>(eng, root_grid, base_level, extra_splits, domain);

        SerialPatchTree<Tvec> sptree(ptree, transform);
        sptree.attach_buf();

        using PtNode = typename SerialPatchTree<Tvec>::PtNode;

        // the box of each patch must be bit-identical to its leaf in the serial tree
        std::unordered_map<u64, PtNode> leaf_nodes;
        sptree.host_for_each_leafs(
            [](u64, PtNode) {
                return true;
            },
            [&](u64 patch_id, PtNode n) {
                leaf_nodes[patch_id] = n;
            });

        REQUIRE_EQUAL(leaf_nodes.size(), leaves.size());

        // AdaptiveCpp compiles with -ffp-contract=fast, and whether a fma is used is decided per
        // call site (and depends on the target cpu). Hence a patch box computed with the patch to
        // object coordinates transform (coord * fact + domain.lower) can differ in f32 from the box
        // of its leaf in the tree by the rounding error of the transform (a fma rounds once
        // instead of twice), and the boxes of neighbouring leaves can overlap by that much.
        using Tscal = shambase::VecComponent<Tvec>;
        Tvec tol;
        for (u32 a = 0; a < 3; a++) {
            tol[a] = std::numeric_limits<Tscal>::epsilon()
                     * (domain.upper[a] - domain.lower[a]
                        + sycl::fmax(sycl::fabs(domain.lower[a]), sycl::fabs(domain.upper[a])));
        }
        auto vec_near = [&](Tvec a, Tvec b) {
            return sycl::fabs(a.x() - b.x()) <= tol.x() && sycl::fabs(a.y() - b.y()) <= tol.y()
                   && sycl::fabs(a.z() - b.z()) <= tol.z();
        };

        // the box of each patch must match the box of its leaf in the serial tree
        u32 box_mismatch = 0;
        u32 box_inexact  = 0;
        for (Patch &p : leaves) {
            auto box  = transform.to_obj_coord(p);
            PtNode &n = leaf_nodes.at(p.id_patch);
            if (!vec_near(box.lower, n.box_min) || !vec_near(box.upper, n.box_max)) {
                box_mismatch++;
            } else if (
                !vec_bit_equal(box.lower, n.box_min) || !vec_bit_equal(box.upper, n.box_max)) {
                box_inexact++;
            }
        }
        if (box_inexact > 0) {
            shamlog_debug_ln(
                "SerialPatchTreeTests",
                "patch boxes not bit-identical to their leaf box (rounding of the transform) :",
                box_inexact);
        }
        REQUIRE_EQUAL(box_mismatch, 0);

        // giving the current patch must not change the result of the full tree search
        u32 n_test_patches = std::min<u32>(leaves.size(), 48);
        u32 mismatch       = 0;
        u32 not_kept       = 0;
        u32 total_moved    = 0;
        u32 total_invalid  = 0;
        u32 not_in_owner   = 0;
        u32 overlap        = 0;
        u32 outside_leaf   = 0;

        // is the position in the box of the leaf linked to patch `id`
        auto in_patch = [&](Tvec p, u64 id) {
            PtNode &n = leaf_nodes.at(id);
            return Patch::is_in_patch_converted(p, n.box_min, n.box_max);
        };

        // same but with the box of the leaf extended by the rounding error of the transform
        auto in_patch_rounding = [&](Tvec p, u64 id) {
            PtNode &n = leaf_nodes.at(id);
            return Patch::is_in_patch_converted(p, n.box_min - tol, n.box_max + tol);
        };

        for (u32 ip = 0; ip < n_test_patches; ip++) {
            Patch cur_p = leaves[ip];
            auto box    = transform.to_obj_coord(cur_p);

            std::vector<Tvec> pos = gen_positions(eng, 2000, box, domain);
            u32 len               = pos.size();

            sham::DeviceBuffer<Tvec> pos_buf(len, dev_sched);
            pos_buf.copy_from_stdvec(pos);

            sycl::buffer<u64> ref_buf = sptree.compute_patch_owner(dev_sched, pos_buf, len);
            sycl::buffer<u64> new_buf = sptree.compute_patch_owner(
                dev_sched, pos_buf, len, cur_p.id_patch, box.lower, box.upper);

            std::vector<u64> ref = buf_to_vec(ref_buf, len);
            std::vector<u64> res = buf_to_vec(new_buf, len);

            for (u32 i = 0; i < len; i++) {
                bool in = Patch::is_in_patch_converted(pos[i], box.lower, box.upper);
                if (!in && ref[i] != cur_p.id_patch) {
                    total_moved++;
                }
                if (in && res[i] != cur_p.id_patch) {
                    not_kept++;
                }
                // As the current patch box computed here (as in
                // ReattributeDataUtility::compute_new_pid) can differ from its tree box by the
                // rounding error of the transform, an object kept by the early return can be up to
                // that rounding error outside of the tree box of its patch, and in an overlap of
                // boxes the tree search returns the first leaf containing the object instead of the
                // current patch.

                // an owner from the tree search must always contain the object
                if (ref[i] != u64_max && !in_patch(pos[i], ref[i])) {
                    not_in_owner++;
                }
                if (in) {
                    // early return : in the given box, and outside of the tree box at most by the
                    // rounding error of the transform
                    if (res[i] == u64_max || !in_patch_rounding(pos[i], res[i])) {
                        not_in_owner++;
                    } else if (!in_patch(pos[i], res[i])) {
                        outside_leaf++;
                    }
                } else if (res[i] != u64_max && !in_patch(pos[i], res[i])) {
                    not_in_owner++;
                }

                // a mismatch is tolerated only in the overlap of the given box and the tree box
                // of the owner found by the tree search
                if (ref[i] != res[i]) {
                    if (in && ref[i] != u64_max && in_patch(pos[i], ref[i])) {
                        overlap++;
                    } else {
                        mismatch++;
                    }
                }
                if (res[i] == u64_max) {
                    total_invalid++;
                }
            }
        }

        if (overlap > 0 || outside_leaf > 0) {
            shamlog_debug_ln(
                "SerialPatchTreeTests",
                "objects kept by the early return outside of the tree box of their patch :",
                outside_leaf,
                ", in an overlap of boxes (owner differs from the tree search) :",
                overlap);
        }

        REQUIRE_EQUAL(mismatch, 0);
        REQUIRE_EQUAL(not_kept, 0);
        REQUIRE_EQUAL(not_in_owner, 0);

        // make sure the test covers both objects changing patch and objects outside the domain
        REQUIRE(total_moved > 0);
        REQUIRE(total_invalid > 0);

        sptree.detach_buf();
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree::compute_patch_owner(current patch)", 1) {
    test_compute_patch_owner<f64_3>({1, 1, 1}, 2, 30, 0x111);
    test_compute_patch_owner<f32_3>({1, 1, 1}, 2, 30, 0x222);
    test_compute_patch_owner<f64_3>({2, 1, 3}, 1, 40, 0x333);
    test_compute_patch_owner<f32_3>({2, 1, 3}, 1, 40, 0x444);
}

namespace {

    /**
     * @brief Median time of compute_patch_owner without (full tree search, as before) and with
     * (current patch first) the current patch
     *
     * The serial tree buffers are recreated before each call as the solver does every step.
     */
    void bench_compute_patch_owner(
        std::string name,
        u32 base_level,
        u32 extra_splits,
        u32 npart,
        f64 moved_frac,
        u32 nrepeat) {

        using Tvec     = f64_3;
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = dev_sched->get_queue();
        std::mt19937 eng(0x666);

        shammath::CoordRange<Tvec> domain{Tvec{-1, -1, -1}, Tvec{1, 1, 1}};
        auto [ptree, leaves, transform, dom]
            = make_test_tree<Tvec>(eng, {1, 1, 1}, base_level, extra_splits, domain);

        SerialPatchTree<Tvec> sptree(ptree, transform);
        u64 tree_bytes = u64(sptree.get_element_count())
                         * (sizeof(typename SerialPatchTree<Tvec>::PtNode) + sizeof(u64));

        // current patch : the largest one, others particles moved slightly outside of it
        Patch cur_p = leaves[0];
        for (Patch &p : leaves) {
            if (p.coord_max[0] - p.coord_min[0] > cur_p.coord_max[0] - cur_p.coord_min[0]) {
                cur_p = p;
            }
        }
        auto box = transform.to_obj_coord(cur_p);
        Tvec ext = (box.upper - box.lower) * 0.05;

        std::vector<Tvec> pos(npart);
        std::uniform_real_distribution<f64> u(0, 1);
        u32 nmoved = 0;
        for (u32 i = 0; i < npart; i++) {
            if (u(eng) < moved_frac) {
                nmoved++;
                // just outside of the lower x face, still in the domain
                Tvec r = rand_in_box(eng, box.lower, box.upper);
                r.x()  = box.lower.x() - ext.x() * u(eng) - 1e-9;
                if (r.x() < domain.lower.x()) {
                    r.x() = box.upper.x() + ext.x() * u(eng);
                }
                pos[i] = r;
            } else {
                pos[i] = rand_in_box(eng, box.lower, box.upper);
            }
        }

        sham::DeviceBuffer<Tvec> pos_buf(npart, dev_sched);
        pos_buf.copy_from_stdvec(pos);

        auto run = [&](bool new_version) -> f64 {
            std::vector<f64> times;
            for (u32 r = 0; r < nrepeat + 1; r++) {
                sptree.attach_buf();

                q.q.wait();
                shambase::Timer t;
                t.start();
                sycl::buffer<u64> buf
                    = (new_version)
                          ? sptree.compute_patch_owner(
                                dev_sched, pos_buf, npart, cur_p.id_patch, box.lower, box.upper)
                          : sptree.compute_patch_owner(dev_sched, pos_buf, npart);
                q.q.wait();
                t.stop();

                sptree.detach_buf();

                if (r > 0) { // first run is warmup
                    times.push_back(t.elapsed_sec());
                }
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        };

        f64 t_old = run(false);
        f64 t_new = run(true);

        logger::raw_ln(
            shambase::format(
                "{:<28} patches={:>7} tree_nodes={:>7} tree_MB={:>7.2f} N={:>8} moved={:>6} "
                "old={:>9.3f} ms new={:>9.3f} ms speedup={:>7.1f}",
                name,
                leaves.size(),
                sptree.get_element_count(),
                f64(tree_bytes) / 1e6,
                npart,
                nmoved,
                t_old * 1e3,
                t_new * 1e3,
                t_old / t_new));

        auto &dat = shamtest::test_data().new_dataset(name);
        dat.add_data("t_old", std::vector<f64>{t_old});
        dat.add_data("t_new", std::vector<f64>{t_new});
        dat.add_data("moved", std::vector<f64>{f64(nmoved)});
        dat.add_data("patches", std::vector<f64>{f64(leaves.size())});
    }

} // namespace

NEW_TEST(Benchmark, "shamrock/scheduler/SerialPatchTree::compute_patch_owner:benchmark", 1) {

    u32 npart = 1u << 22;

    // 8^2 = 64 patches
    bench_compute_patch_owner("small tree, no move", 2, 0, npart, 0, 10);
    bench_compute_patch_owner("small tree, 0.1% move", 2, 0, npart, 1e-3, 10);

    // 8^6 + 77000 * 7 ~ 800k patches
    bench_compute_patch_owner("800k patches, no move", 6, 77000, npart, 0, 10);
    bench_compute_patch_owner("800k patches, 0.1% move", 6, 77000, npart, 1e-3, 10);
}

namespace {

    constexpr u64 max_coord = shamrock::scheduler::HilbertLoadBalance<u64>::max_box_sz;

    /// Give every leaf a patch id and every internal node u64_max
    void link_leafs(PatchTree &ptree) {
        u64 pid = 0;
        for (auto &[key, node] : ptree.tree) {
            node.linked_patchid = (node.tree_node.is_leaf) ? pid++ : u64_max;
        }
    }

    /// Uniform octree: a single root split `level` times
    PatchTree make_uniform_tree(u32 level) {
        PatchTree ptree;
        ptree.insert_root_node(
            0, shamrock::patch::PatchCoord<3>({0, 0, 0}, {max_coord, max_coord, max_coord}));
        for (u32 l = 0; l < level; l++) {
            std::vector<u64> leafs(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : leafs) {
                ptree.split_node(id);
            }
        }
        link_leafs(ptree);
        return ptree;
    }

    f64 median(std::vector<f64> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

} // namespace

/// Median time of the SerialPatchTree<f64_3> build and of attach_buf, as done every step by the
/// solvers, on uniform octrees up to 2M patches
NEW_TEST(Benchmark, "shamrock/scheduler/SerialPatchTree:benchmark", 1) {
    constexpr u32 rep_count = 7;

    shamrock::patch::PatchCoordTransform<f64_3> transform(
        {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}},
        {{-0.6, -0.6, -0.6}, {0.6, 0.6, 0.6}});

    std::vector<f64> patch_counts, node_counts, t_build, t_attach_buf;

    for (u32 level = 1; level <= 7; level++) {
        PatchTree ptree = make_uniform_tree(level);

        std::vector<f64> rep_build, rep_attach_buf;
        for (u32 rep = 0; rep < rep_count; rep++) {
            shambase::Timer timer_build;
            timer_build.start();
            SerialPatchTree<f64_3> sptree(ptree, transform);
            timer_build.stop();

            shambase::Timer timer_attach_buf;
            timer_attach_buf.start();
            sptree.attach_buf();
            timer_attach_buf.stop();

            rep_build.push_back(timer_build.elapsed_sec());
            rep_attach_buf.push_back(timer_attach_buf.elapsed_sec());
        }

        patch_counts.push_back(ptree.leaf_key.size());
        node_counts.push_back(ptree.tree.size());
        t_build.push_back(median(rep_build));
        t_attach_buf.push_back(median(rep_attach_buf));
    }

    std::string out = "SerialPatchTree build :\n";

    for (u32 i = 0; i < patch_counts.size(); i++) {
        out += sham::format(
            "patches = {:8} nodes = {:8} build = {:9.3f} ms "
            "attach_buf = {:.3f} ms\n",
            patch_counts[i],
            node_counts[i],
            t_build[i] * 1e3,
            t_attach_buf[i] * 1e3);
    }

    logger::info_ln("ShamrockTest", out);

    auto &dataset = shamtest::test_data().new_dataset("SerialPatchTree build");
    dataset.add_data("Npatch", patch_counts);
    dataset.add_data("Nnode", node_counts);
    dataset.add_data("t_build", t_build);
    dataset.add_data("t_attach_buf", t_attach_buf);
}

namespace {

    /**
     * @brief Random patch tree : `nroots` roots side by side along x, all split `base_level`
     * times, then `nops` random splits of a leaf or merges of a parent of only leafs (merges leave
     * holes in the node keys)
     */
    PatchTree make_random_tree(std::mt19937 &eng, u32 nroots, u32 base_level, u32 nops) {
        PatchTree ptree;

        u64 root_width = (max_coord + 1) / nroots;
        for (u32 i = 0; i < nroots; i++) {
            ptree.insert_root_node(
                i,
                shamrock::patch::PatchCoord<3>(
                    {root_width * i, 0, 0}, {root_width * (i + 1) - 1, max_coord, max_coord}));
        }

        for (u32 l = 0; l < base_level; l++) {
            std::vector<u64> leafs(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : leafs) {
                ptree.split_node(id);
            }
        }

        auto pick = [&](const std::unordered_set<u64> &set) {
            std::vector<u64> keys(set.begin(), set.end());
            std::sort(keys.begin(), keys.end());
            return keys[std::uniform_int_distribution<u64>(0, keys.size() - 1)(eng)];
        };

        std::uniform_int_distribution<u32> op(0, 9);
        for (u32 i = 0; i < nops; i++) {
            if (op(eng) < 3 && !ptree.parent_of_only_leaf_key.empty()) {
                ptree.merge_node_dm1(pick(ptree.parent_of_only_leaf_key));
            } else {
                ptree.split_node(pick(ptree.leaf_key));
            }
        }

        link_leafs(ptree);
        return ptree;
    }

    /**
     * @brief Check that a SerialPatchTree is an exact image of the PatchTree it was built from,
     * whatever the ordering of its nodes : walk both trees from matching roots and compare the
     * boxes, linked patch ids and childs (in order)
     */
    template<class Tvec>
    void check_serial_tree(PatchTree &ptree) {
        using Tscal  = shambase::VecComponent<Tvec>;
        using PtNode = typename SerialPatchTree<Tvec>::PtNode;

        shammath::CoordRange<Tvec> domain{Tvec{-0.6, -0.3, 0.1}, Tvec{0.7, 0.4, 1.2}};
        shamrock::patch::PatchCoordTransform<Tvec> transform(
            {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}}, domain);

        SerialPatchTree<Tvec> sptree(ptree, transform);
        sptree.attach_buf();

        REQUIRE_EQUAL(sptree.get_element_count(), ptree.tree.size());
        REQUIRE_EQUAL(sptree.root_count, ptree.roots_id.size());
        REQUIRE_EQUAL(sptree.get_roots_ids().size(), ptree.roots_id.size());

        // the boxes may differ by the rounding error of the transform (fma contraction can
        // differ between call sites, see the compute_patch_owner test)
        Tvec tol;
        for (u32 a = 0; a < 3; a++) {
            tol[a] = 4 * std::numeric_limits<Tscal>::epsilon()
                     * (domain.upper[a] - domain.lower[a]
                        + sycl::fmax(sycl::fabs(domain.lower[a]), sycl::fabs(domain.upper[a])));
        }
        auto vec_near = [&](Tvec a, Tvec b) {
            return sycl::fabs(a.x() - b.x()) <= tol.x() && sycl::fabs(a.y() - b.y()) <= tol.y()
                   && sycl::fabs(a.z() - b.z()) <= tol.z();
        };

        sycl::host_accessor tree{shambase::get_check_ref(sptree.serial_tree_buf), sycl::read_only};
        sycl::host_accessor lpid{
            shambase::get_check_ref(sptree.linked_patch_ids_buf), sycl::read_only};

        // (key in the patch tree, id in the serial tree)
        std::vector<std::pair<u64, u64>> stack;
        u32 iroot = 0;
        for (u64 root_key : ptree.roots_id) {
            stack.push_back({root_key, sptree.get_roots_ids()[iroot++]});
        }

        std::vector<bool> visited(sptree.get_element_count(), false);
        u32 max_level        = 0;
        u64 bad_id           = 0;
        u64 visited_twice    = 0;
        u64 box_mismatch     = 0;
        u64 lpid_mismatch    = 0;
        u64 is_leaf_mismatch = 0;

        while (!stack.empty()) {
            auto [key, id] = stack.back();
            stack.pop_back();

            if (id >= visited.size()) {
                bad_id++;
                continue;
            }
            if (visited[id]) {
                visited_twice++;
                continue;
            }
            visited[id] = true;

            const PatchTree::Node &node = ptree.tree.at(key);
            PtNode snode                = tree[id];
            max_level                   = std::max(max_level, node.tree_node.level);

            auto box = transform.to_obj_coord(node.patch_coord.get_patch_range());
            if (!vec_near(box.lower, snode.box_min) || !vec_near(box.upper, snode.box_max)) {
                box_mismatch++;
            }
            if (lpid[id] != node.linked_patchid) {
                lpid_mismatch++;
            }
            if (node.tree_node.is_leaf != (snode.childs_id[0] == u64_max)) {
                is_leaf_mismatch++;
                continue;
            }
            if (!node.tree_node.is_leaf) {
                for (u32 c = 0; c < 8; c++) {
                    stack.push_back({node.tree_node.childs_nid[c], snode.childs_id[c]});
                }
            }
        }

        REQUIRE_EQUAL(bad_id, 0);
        REQUIRE_EQUAL(visited_twice, 0);
        REQUIRE_EQUAL(box_mismatch, 0);
        REQUIRE_EQUAL(lpid_mismatch, 0);
        REQUIRE_EQUAL(is_leaf_mismatch, 0);
        REQUIRE_EQUAL(std::count(visited.begin(), visited.end(), true), visited.size());
        REQUIRE_EQUAL(sptree.get_level_count(), max_level);
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree:build", 1) {
    std::mt19937 eng(0x5e71a1);

    // nroots, base_level, nops
    std::vector<std::array<u32, 3>> cases = {
        {1, 0, 0},
        {1, 2, 200},
        {2, 1, 400},
        {4, 2, 300},
        // large enough to go through the OpenMP paths of the build
        {4, 5, 64},
    };

    for (auto [nroots, base_level, nops] : cases) {
        PatchTree ptree = make_random_tree(eng, nroots, base_level, nops);
        check_serial_tree<f64_3>(ptree);
        check_serial_tree<f32_3>(ptree);
    }
}
