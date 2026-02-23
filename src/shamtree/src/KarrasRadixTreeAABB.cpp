// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file KarrasRadixTreeAABB.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shamtree/KarrasRadixTreeAABB.hpp"

namespace shamtree {
    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> prepare_karras_radix_tree_aabb(
        const KarrasRadixTree &tree, KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb) {

        KarrasRadixTreeAABB<Tvec> ret = std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb);

        ret.buf_aabb_min.resize(tree.get_total_cell_count());
        ret.buf_aabb_max.resize(tree.get_total_cell_count());

        return ret;
    }

    template<class Tvec>
    void propagate_aabb_up(KarrasRadixTreeAABB<Tvec> &tree_aabb, const KarrasRadixTree &tree) {

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 int_cell_count = tree.get_internal_cell_count();

        if (int_cell_count == 0) {
            return;
        }

        auto step = [&]() {
            auto traverser = tree.get_structure_traverser();

            sham::kernel_call(
                q,
                sham::MultiRef{traverser},
                sham::MultiRef{tree_aabb.buf_aabb_min, tree_aabb.buf_aabb_max},
                int_cell_count,
                [=](u32 gid,
                    auto tree_traverser,
                    Tvec *__restrict cell_min,
                    Tvec *__restrict cell_max) {
                    u32 left_child  = tree_traverser.get_left_child(gid);
                    u32 right_child = tree_traverser.get_right_child(gid);

                    Tvec bminl = cell_min[left_child];
                    Tvec bminr = cell_min[right_child];
                    Tvec bmaxl = cell_max[left_child];
                    Tvec bmaxr = cell_max[right_child];

                    Tvec bmin = sham::min(bminl, bminr);
                    Tvec bmax = sham::max(bmaxl, bmaxr);

                    cell_min[gid] = bmin;
                    cell_max[gid] = bmax;
                });
        };

        for (u32 i = 0; i < tree.tree_depth; i++) {
            step();
        }
    }

    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb(
        const KarrasRadixTree &tree,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        const std::function<void(KarrasRadixTreeAABB<Tvec> &, u32)> &fct_fill_leaf) {

        auto aabbs = prepare_karras_radix_tree_aabb(
            tree, std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb));

        fct_fill_leaf(aabbs, tree.get_internal_cell_count());

        propagate_aabb_up(aabbs, tree);

        return aabbs;
    }

    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb_from_positions(
        const KarrasRadixTree &tree,
        const LeafCellIterator &cell_it,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        sham::DeviceBuffer<Tvec> &positions) {
        __shamrock_stack_entry();

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        auto fill_leafs = [&](KarrasRadixTreeAABB<Tvec> &tree_aabb, u32 leaf_offset) {
            sham::kernel_call(
                q,
                sham::MultiRef{positions, cell_it},
                sham::MultiRef{tree_aabb.buf_aabb_min, tree_aabb.buf_aabb_max},
                tree.get_leaf_count(),
                [leaf_offset](
                    u32 i, const Tvec *pos, auto cell_iter, Tvec *comp_min, Tvec *comp_max) {
                    // how to lose a f****ing afternoon :
                    // 1 - code a nice algorithm that should optimize the code
                    // 2 - pass all the tests
                    // 3 - benchmark it and discover big loss in perf for no reasons
                    // 4 - change a parameter and discover a segfault (on GPU to have more fun ....)
                    // 5 - find that actually the core algorithm of the code create a bug in the new
                    // thing 6 - discover that every value in everything is wrong 7 - spent the
                    // whole night on it 8 - start putting prints everywhere 9 - isolate a bugged id
                    // 10 - try to understand why a f***ing leaf is as big as the root of the tree
                    // 11 - **** a few hours latter 12 - the goddam c++ standard define
                    // std::numeric_limits<float>::min() to be epsilon instead of -max 13 - road
                    // rage 14
                    // - open a bier alt f4 the ide
                    Tvec min = shambase::VectorProperties<Tvec>::get_max();
                    Tvec max = -shambase::VectorProperties<Tvec>::get_max();

                    cell_iter.for_each_in_leaf_cell(i, [&](u32 obj_id) {
                        Tvec r = pos[obj_id];

                        min = sham::min(min, r);
                        max = sham::max(max, r);
                    });

                    comp_min[leaf_offset + i] = min;
                    comp_max[leaf_offset + i] = max;
                });
        };

        return compute_tree_aabb<Tvec>(
            tree, std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb), fill_leafs);
    }

    template<class Tvec>
    KarrasRadixTreeAABB<Tvec> compute_tree_aabb_from_position_ranges(
        const KarrasRadixTree &tree,
        const LeafCellIterator &cell_it,
        KarrasRadixTreeAABB<Tvec> &&recycled_tree_aabb,
        sham::DeviceBuffer<Tvec> &min,
        sham::DeviceBuffer<Tvec> &max) {
        __shamrock_stack_entry();
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        auto fill_leafs = [&](KarrasRadixTreeAABB<Tvec> &tree_aabb, u32 leaf_offset) {
            sham::kernel_call(
                q,
                sham::MultiRef{min, max, cell_it},
                sham::MultiRef{tree_aabb.buf_aabb_min, tree_aabb.buf_aabb_max},
                tree.get_leaf_count(),
                [leaf_offset](
                    u32 i,
                    const Tvec *min_pos,
                    const Tvec *max_pos,
                    auto cell_iter,
                    Tvec *comp_min,
                    Tvec *comp_max) {
                    Tvec min = shambase::VectorProperties<Tvec>::get_max();
                    Tvec max = -shambase::VectorProperties<Tvec>::get_max();

                    cell_iter.for_each_in_leaf_cell(i, [&](u32 obj_id) {
                        Tvec r_min = min_pos[obj_id];
                        Tvec r_max = max_pos[obj_id];

                        min = sham::min(min, r_min);
                        max = sham::max(max, r_max);
                    });

                    comp_min[leaf_offset + i] = min;
                    comp_max[leaf_offset + i] = max;
                });
        };

        return compute_tree_aabb<Tvec>(
            tree, std::forward<KarrasRadixTreeAABB<Tvec>>(recycled_tree_aabb), fill_leafs);
    }

} // namespace shamtree

#ifndef DOXYGEN
template shamtree::KarrasRadixTreeAABB<f64_3> shamtree::prepare_karras_radix_tree_aabb<f64_3>(
    const KarrasRadixTree &tree, shamtree::KarrasRadixTreeAABB<f64_3> &&recycled_tree_aabb);

template void shamtree::propagate_aabb_up<f64_3>(
    shamtree::KarrasRadixTreeAABB<f64_3> &tree_aabb, const KarrasRadixTree &tree);

template shamtree::KarrasRadixTreeAABB<f64_3> shamtree::compute_tree_aabb<f64_3>(
    const KarrasRadixTree &tree,
    shamtree::KarrasRadixTreeAABB<f64_3> &&recycled_tree_aabb,
    const std::function<void(shamtree::KarrasRadixTreeAABB<f64_3> &, u32)> &fct_fill_leaf);

template shamtree::KarrasRadixTreeAABB<f64_3> shamtree::compute_tree_aabb_from_positions<f64_3>(
    const KarrasRadixTree &tree,
    const LeafCellIterator &cell_it,
    shamtree::KarrasRadixTreeAABB<f64_3> &&recycled_tree_aabb,
    sham::DeviceBuffer<f64_3> &positions);

template shamtree::KarrasRadixTreeAABB<f64_3> shamtree::compute_tree_aabb_from_position_ranges<
    f64_3>(
    const KarrasRadixTree &tree,
    const LeafCellIterator &cell_it,
    shamtree::KarrasRadixTreeAABB<f64_3> &&recycled_tree_aabb,
    sham::DeviceBuffer<f64_3> &min,
    sham::DeviceBuffer<f64_3> &max);
#endif
