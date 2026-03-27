// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/MortonKernels.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeStructureWalker.hpp"
#include "tests/shamrock/tree/TreeTests.hpp"

auto get_Nmax = []() -> f64 {
    return 1e8 * 1;
};

template<class morton_mode, class flt, u32 reduc_lev>
inline void test_tree_build_steps(std::string dset_name) {

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = get_Nmax();

    u32 Nmax = u32(sycl::fmin(Nmax_flt, 2e9));

    auto coord_range = get_test_coord_ranges<vec>();

    auto pos
        = shamalgs::random::mock_buffer_ptr<vec>(0x111, Nmax, coord_range.lower, coord_range.upper);

    shamalgs::memory::move_buffer_on_queue(shamsys::instance::get_compute_queue(), *pos);

    std::vector<f64> times_morton;
    std::vector<f64> times_reduc;
    std::vector<f64> times_karras;
    std::vector<f64> times_compute_int_range;
    std::vector<f64> times_compute_coord_range;
    std::vector<f64> times_morton_build;
    std::vector<f64> times_trailling_fill;
    std::vector<f64> times_index_gen;
    std::vector<f64> times_morton_sort;
    std::vector<f64> times_full_tree;

    std::vector<f64> Npart;

    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        Npart.push_back(u32(cnt));
    }

    for (f64 cnt : Npart) {
        times_morton.push_back(0);
        times_reduc.push_back(0);
        times_karras.push_back(0);
        times_compute_int_range.push_back(0);
        times_compute_coord_range.push_back(0);
        times_morton_build.push_back(0);
        times_trailling_fill.push_back(0);
        times_index_gen.push_back(0);
        times_morton_sort.push_back(0);
        times_full_tree.push_back(0);
    }

    auto get_repetition_count = [](f64 cnt) {
        if (cnt < 1e5)
            return 100;
        return 30;
    };

    u32 index = 0;
    for (f64 cnt : Npart) {
        shamlog_debug_ln("TestTreePerf", cnt, dset_name);
        for (u32 rep_count = 0; rep_count < get_repetition_count(cnt); rep_count++) {

            shambase::Timer timer;
            u32 cnt_obj = cnt;

            auto time_func = [](auto f) {
                shamsys::instance::get_compute_queue().wait();
                shambase::Timer timer;
                timer.start();

                f();
                shamsys::instance::get_compute_queue().wait();

                timer.end();
                return timer.nanosec / 1.e9;
            };

            {
                shamrock::tree::TreeMortonCodes<morton_mode> tree_morton_codes;
                shamrock::tree::TreeReducedMortonCodes<morton_mode> tree_reduced_morton_codes;
                shamrock::tree::TreeStructure<morton_mode> tree_struct;

                times_morton[index] += (time_func([&]() {
                    tree_morton_codes.build(
                        shamsys::instance::get_compute_queue(), coord_range, cnt_obj, *pos);
                }));

                bool one_cell_mode;
                times_reduc[index] += (time_func([&]() {
                    tree_reduced_morton_codes.build(
                        shamsys::instance::get_compute_queue(),
                        cnt_obj,
                        reduc_lev,
                        tree_morton_codes,
                        one_cell_mode);
                }));

                times_karras[index] += (time_func([&]() {
                    if (!one_cell_mode) {
                        tree_struct.build(
                            shamsys::instance::get_compute_queue(),
                            tree_reduced_morton_codes.tree_leaf_count - 1,
                            *tree_reduced_morton_codes.buf_tree_morton);
                    } else {
                        tree_struct.build_one_cell_mode();
                    }
                }));
            }

            {
                RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
                    shamsys::instance::get_compute_queue(),
                    {coord_range.lower, coord_range.upper},
                    pos,
                    cnt,
                    reduc_lev);

                times_compute_int_range[index] += (time_func([&]() {
                    rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                }));

                shamsys::instance::get_compute_queue().wait();
                times_compute_coord_range[index] += (time_func([&]() {
                    rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
                }));
            }

            {

                using namespace shamrock::sfc;

                u32 morton_len = sham::roundup_pow2_clz(cnt_obj);

                auto out_buf_morton = std::make_unique<sycl::buffer<morton_mode>>(morton_len);

                times_morton_build[index] += (time_func([&]() {
                    MortonKernels<morton_mode, vec, 3>::sycl_xyz_to_morton(
                        shamsys::instance::get_compute_queue(),
                        cnt_obj,
                        *pos,
                        coord_range.lower,
                        coord_range.upper,
                        out_buf_morton);
                }));

                times_trailling_fill[index] += (time_func([&]() {
                    MortonKernels<morton_mode, vec, 3>::sycl_fill_trailling_buffer(
                        shamsys::instance::get_compute_queue(),
                        cnt_obj,
                        morton_len,
                        out_buf_morton);
                }));

                std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;

                times_index_gen[index] += (time_func([&]() {
                    out_buf_particle_index_map
                        = std::make_unique<sycl::buffer<u32>>(shamalgs::algorithm::gen_buffer_index(
                            shamsys::instance::get_compute_queue(), morton_len));
                }));

                times_morton_sort[index] += (time_func([&]() {
                    sycl_sort_morton_key_pair(
                        shamsys::instance::get_compute_queue(),
                        morton_len,
                        out_buf_particle_index_map,
                        out_buf_morton);
                }));
            }

            {
                shamsys::instance::get_compute_queue().wait();
                shambase::Timer timer2;
                timer2.start();

                RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
                    shamsys::instance::get_compute_queue(),
                    {coord_range.lower, coord_range.upper},
                    pos,
                    cnt,
                    reduc_lev);

                rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
                shamsys::instance::get_compute_queue().wait();
                timer2.end();
                times_full_tree[index] += (timer2.nanosec / 1.e9);
            }
        }

        index++;
    }

    index = 0;
    for (f64 cnt : Npart) {

        times_morton[index] /= get_repetition_count(cnt);
        times_reduc[index] /= get_repetition_count(cnt);
        times_karras[index] /= get_repetition_count(cnt);
        times_compute_int_range[index] /= get_repetition_count(cnt);
        times_compute_coord_range[index] /= get_repetition_count(cnt);
        times_morton_build[index] /= get_repetition_count(cnt);
        times_trailling_fill[index] /= get_repetition_count(cnt);
        times_index_gen[index] /= get_repetition_count(cnt);
        times_morton_sort[index] /= get_repetition_count(cnt);
        times_full_tree[index] /= get_repetition_count(cnt);

        index++;
    }

    auto &dset = shamtest::test_data().new_dataset(dset_name);

    dset.add_data("Npart", Npart);

    dset.add_data("times_morton", times_morton);
    dset.add_data("times_reduc", times_reduc);
    dset.add_data("times_karras", times_karras);
    dset.add_data("times_compute_int_range", times_compute_int_range);
    dset.add_data("times_compute_coord_range", times_compute_coord_range);
    dset.add_data("times_morton_build", times_morton_build);
    dset.add_data("times_trailling_fill", times_trailling_fill);
    dset.add_data("times_index_gen", times_index_gen);
    dset.add_data("times_morton_sort", times_morton_sort);
    dset.add_data("times_full_tree", times_full_tree);
}

template<class u_morton, class flt>
class SPHTestInteractionCrit {
    using vec = sycl::vec<flt, 3>;

    public:
    RadixTree<u_morton, vec> &tree;
    sycl::buffer<vec> &positions;
    u32 part_count;
    flt Rpart;

    class Access {
        public:
        sycl::accessor<vec, 1, sycl::access::mode::read> part_pos;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_min;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_max;

        flt Rpart;
        flt Rpart_pow2;

        Access(SPHTestInteractionCrit crit, sycl::handler &cgh)
            : part_pos{crit.positions, cgh, sycl::read_only}, Rpart(crit.Rpart),
              Rpart_pow2(crit.Rpart * crit.Rpart),
              tree_cell_coordrange_min{
                  *crit.tree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only},
              tree_cell_coordrange_max{
                  *crit.tree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only} {}

        class ObjectValues {
            public:
            vec xyz_a;
            ObjectValues(Access acc, u32 index) : xyz_a(acc.part_pos[index]) {}
        };
    };

    inline static bool criterion(
        u32 node_index, Access acc, typename Access::ObjectValues current_values) {
        vec cur_pos_min_cell_b = acc.tree_cell_coordrange_min[node_index];
        vec cur_pos_max_cell_b = acc.tree_cell_coordrange_max[node_index];

        vec box_int_sz = {acc.Rpart, acc.Rpart, acc.Rpart};

        return BBAA::cella_neigh_b(
                   current_values.xyz_a - box_int_sz,
                   current_values.xyz_a + box_int_sz,
                   cur_pos_min_cell_b,
                   cur_pos_max_cell_b)
               || BBAA::cella_neigh_b(
                   current_values.xyz_a,
                   current_values.xyz_a,
                   cur_pos_min_cell_b - box_int_sz,
                   cur_pos_min_cell_b + box_int_sz);
    };
};

template<class morton_mode, class flt, u32 reduc_lev>
void test_sph_iter_overhead(std::string dset_name) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    // setup the particle distribution

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = get_Nmax();

    u32 Nmax = 2U << 23U;

    auto coord_range = get_test_coord_ranges<vec>();

    std::vector<f64> Npart;
    std::vector<f64> avg_neigh;
    std::vector<f64> var_neigh;
    std::vector<f64> rpart_vec;
    std::vector<f64> times;

    auto mix_seed = [](f64 seed) -> u32 {
        f64 a = 16807;
        f64 m = 2147483647;
        seed  = std::fmod((a * seed), m);
        return u32_max * (seed / m);
    };

    u32 test_per_n = 10;
    u32 seed       = 0x111;
    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        for (u32 i = 0; i < 15; i++) {
            seed        = mix_seed(seed);
            u32 len_pos = cnt;

            flt volume_per_obj = coord_range.get_volume() / len_pos;

            flt len_per_obj = sycl::cbrt(volume_per_obj);

            flt rpart = 0;
            {
                std::mt19937 eng(seed);
                rpart = std::uniform_real_distribution<flt>(0, len_per_obj * 4)(eng);
            }

            shamlog_debug_ln(
                "TestTreePerf",
                shambase::format(
                    "dataset : {}, len={:e} seed={:10} len_p_obj={:e} rpart={:e}",
                    dset_name,
                    f32(len_pos),
                    seed,
                    len_per_obj,
                    rpart));

            auto pos = shamalgs::random::mock_buffer_ptr<vec>(
                seed, len_pos, coord_range.lower, coord_range.upper);

            sycl::buffer<u32> neighbours(len_pos);

            // try{
            RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
                shamsys::instance::get_compute_queue(),
                {coord_range.lower, coord_range.upper},
                pos,
                cnt,
                reduc_lev);

            rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            rtree.convert_bounding_box(shamsys::instance::get_compute_queue());

            auto benchmark = [&]() -> f64 {
                shambase::Timer t;

                q.wait();
                t.start();

                using Criterion    = SPHTestInteractionCrit<morton_mode, flt>;
                using CriterionAcc = typename Criterion::Access;
                using CriterionVal = typename CriterionAcc::ObjectValues;

                using namespace shamrock::tree;
                TreeStructureWalker walk = generate_walk<Recompute>(
                    rtree.tree_struct,
                    len_pos,
                    SPHTestInteractionCrit<morton_mode, flt>{rtree, *pos, len_pos, rpart});

                q.submit([&](sycl::handler &cgh) {
                    auto walker        = walk.get_access(cgh);
                    auto leaf_iterator = rtree.get_leaf_access(cgh);

                    sycl::accessor neigh_count{neighbours, cgh, sycl::write_only, sycl::no_init};

                    cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
                        u32 sum = 0;

                        CriterionVal int_values{
                            walker.criterion(), static_cast<u32>(item.get_linear_id())};

                        walker.for_each_node(
                            item,
                            int_values,
                            [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                                leaf_iterator.iter_object_in_leaf(
                                    leaf_iterator_id, [&](u32 obj_id) {
                                        vec xyz_b = walker.criterion().part_pos[obj_id];
                                        vec dxyz  = xyz_b - int_values.xyz_a;
                                        flt dot_  = sycl::dot(dxyz, dxyz);

                                        if (dot_ < walker.criterion().Rpart_pow2) {
                                            sum += 1;
                                        }
                                    });
                            },
                            [&](u32 node_id) {});

                        neigh_count[item] = sum;
                    });
                });

                q.wait();
                t.end();

                return t.nanosec * 1e-9;
            };

            f64 time = benchmark();

            {

                f64 npart     = len_pos;
                f64 neigh_avg = 0;
                f64 neigh_var = 0;

                {
                    sycl::host_accessor acc{neighbours, sycl::read_only};
                    for (u32 i = 0; i < len_pos; i++) {
                        neigh_avg += acc[i];
                    }
                    neigh_avg /= len_pos;

                    for (u32 i = 0; i < len_pos; i++) {
                        neigh_var += (acc[i] - neigh_avg) * (acc[i] - neigh_avg);
                    }
                    neigh_var /= len_pos;
                }

                Npart.push_back(npart);
                avg_neigh.push_back(neigh_avg);
                var_neigh.push_back(neigh_var);
                times.push_back(time);
                rpart_vec.push_back(rpart);
            }
            //}catch(...){
            //    continue;
            //}
        }
    }

    auto &dat_test = shamtest::test_data().new_dataset(dset_name);

    dat_test.add_data("Nobj", Npart);
    dat_test.add_data("avg_neigh", avg_neigh);
    dat_test.add_data("var_neigh", var_neigh);
    dat_test.add_data("time", times);
    dat_test.add_data("rpart", rpart_vec);
}

template<class T>
using buf_access_read = sycl::accessor<T, 1, sycl::access::mode::read, sycl::target::device>;
template<class T>
using buf_access_read_write
    = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::target::device>;

template<class morton_mode, u32 reduc_lev, class FctRho>
f64 amr_walk_perf(
    f64 lambda_tilde,
    std::vector<f64> &Npart,
    std::vector<f64> &avg_neigh,
    std::vector<f64> &var_neigh,
    std::vector<f64> &time_refine,
    std::vector<f64> &time_tree,
    std::vector<f64> &time_walk) {

    using namespace shamrock::patch;
    using namespace shamrock::scheduler;

    std::shared_ptr<PatchDataLayout> layout_ptr
        = std::make_shared<PatchDataLayout>(std::vector<std::string>{"main"});
    auto &layout = layout_ptr->get_layer_ref(0);
    layout.add_field<u64_3>("cell_min", 1);
    layout.add_field<u64_3>("cell_max", 1);
    PatchScheduler sched(layout_ptr, 1e9, 1);

    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;

    Grid grid(sched);

    u64 base_cell_size  = 1U << 20U;
    u32 base_cell_count = 1U << 4U;
    u64 int_box_len     = base_cell_size * base_cell_count;

    shammath::CoordRange<u64_3> base_range({0, 0, 0}, {int_box_len, int_box_len, int_box_len});
    shammath::CoordRange<f32_3> real_coord_range(
        {
            -1,
            -1,
            -1,
        },
        {1, 1, 1});

    grid.make_base_grid(
        {0, 0, 0},
        {base_cell_size, base_cell_size, base_cell_size},
        {base_cell_count, base_cell_count, base_cell_count});

    class RefineCritCellAccessor {
        public:
        const u64_3 *cell_low_bound;
        const u64_3 *cell_high_bound;

        shammath::CoordRangeTransform<u64_3, f32_3> transform;

        RefineCritCellAccessor(
            sham::EventList &depends_list,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            shammath::CoordRange<u64_3> base_range,
            shammath::CoordRange<f32_3> real_coord_range)
            : transform(base_range, real_coord_range) {

            auto &buf_cell_low_bound  = pdat.get_field<u64_3>(0).get_buf();
            auto &buf_cell_high_bound = pdat.get_field<u64_3>(1).get_buf();
            cell_low_bound            = buf_cell_low_bound.get_read_access(depends_list);
            cell_high_bound           = buf_cell_high_bound.get_read_access(depends_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            shammath::CoordRange<u64_3> base_range,
            shammath::CoordRange<f32_3> real_coord_range) {

            auto &buf_cell_low_bound  = pdat.get_field<u64_3>(0).get_buf();
            auto &buf_cell_high_bound = pdat.get_field<u64_3>(1).get_buf();
            buf_cell_low_bound.complete_event_state(resulting_events);
            buf_cell_high_bound.complete_event_state(resulting_events);
        }
    };

    class RefineCellAccessor {
        public:
        RefineCellAccessor(sham::EventList &depends_list, shamrock::patch::PatchDataLayer &pdat) {}

        void finalize(sham::EventList &resulting_events, shamrock::patch::PatchDataLayer &pdat) {}
    };

    sycl::queue &q = shamsys::instance::get_compute_queue();
    q.wait();
    shambase::Timer t_refine;
    t_refine.start();

    bool rerefine = false;

    do {

        f32 lambda_tilde_f32 = lambda_tilde;

        auto splits = grid.gen_refine_list<RefineCritCellAccessor>(
            [lambda_tilde_f32](u32 cell_id, RefineCritCellAccessor acc) -> u32 {
                u64_3 low_bound  = acc.cell_low_bound[cell_id];
                u64_3 high_bound = acc.cell_high_bound[cell_id];

                using namespace shammath;

                CoordRange<f32_3> cell_coords
                    = acc.transform.transform(CoordRange<u64_3>{low_bound, high_bound});

                f32_3 cell_center = (cell_coords.lower + cell_coords.upper) / 2.F;

                f32 Rpow2 = sycl::dot(cell_center, cell_center);

                f32 density       = FctRho::rho(Rpow2);
                f32 lambda_tilde  = lambda_tilde_f32;
                f32 cell_len_side = cell_coords.delt().x();

                // mocked jeans length
                bool should_refine = cell_len_side > (lambda_tilde / sycl::sqrt(density));

                should_refine = should_refine && (high_bound.x() - low_bound.x() > 1);
                should_refine = should_refine && (high_bound.y() - low_bound.y() > 1);
                should_refine = should_refine && (high_bound.z() - low_bound.z() > 1);

                return should_refine;
            },
            base_range,
            real_coord_range);

        rerefine = grid.get_process_refine_count(splits) > 0;

        grid.apply_splits<RefineCellAccessor>(
            std::move(splits),

            [](u32 cur_idx,
               Grid::CellCoord cur_coords,
               std::array<u32, 8> new_cells,
               std::array<Grid::CellCoord, 8> new_cells_coords,
               RefineCellAccessor acc) {}

        );

        if (rerefine) {
            logger::info_ln("TestAMRWalk", "rerefining ...");
        }

    } while (rerefine);

    q.wait();
    t_refine.end();

    PatchData &pdat = sched.patch_data.get_pdat(0);
    Patch p         = sched.patch_list.global.at(0);

    u32 len_pos = pdat.get_obj_cnt();

    logger::info_ln("TestAmrWalk", "obj count :", len_pos);

    class InteractionCrit {
        public:
        shammath::CoordRange<u64_3> bounds;

        RadixTree<u64, u64_3> &tree;
        PatchData &pdat;

        sycl::buffer<u64_3> buf_cell_low_bound;
        sycl::buffer<u64_3> buf_cell_high_bound;

        class Access {
            public:
            sycl::accessor<u64_3, 1, sycl::access::mode::read> cell_low_bound;
            sycl::accessor<u64_3, 1, sycl::access::mode::read> cell_high_bound;

            sycl::accessor<u64_3, 1, sycl::access::mode::read> tree_cell_coordrange_min;
            sycl::accessor<u64_3, 1, sycl::access::mode::read> tree_cell_coordrange_max;

            Access(InteractionCrit crit, sycl::handler &cgh)
                : cell_low_bound{crit.buf_cell_low_bound, cgh, sycl::read_only},
                  cell_high_bound{crit.buf_cell_high_bound, cgh, sycl::read_only},
                  tree_cell_coordrange_min{
                      *crit.tree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only},
                  tree_cell_coordrange_max{
                      *crit.tree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only} {}

            class ObjectValues {
                public:
                shammath::AABB<u64_3> cell_bound;
                ObjectValues(Access acc, u32 index)
                    : cell_bound(acc.cell_low_bound[index], acc.cell_high_bound[index]) {}
            };
        };

        static bool criterion(
            u32 node_index, Access acc, typename Access::ObjectValues current_values) {

            shammath::AABB<u64_3> tree_cell_bound{
                acc.tree_cell_coordrange_min[node_index], acc.tree_cell_coordrange_max[node_index]};

            shammath::AABB<u64_3> intersect
                = tree_cell_bound.get_intersect(current_values.cell_bound);

            // logger::raw(
            //     shambase::format("{} {} | {} {} -> {} {} -> {}\n",
            //     tree_cell_bound.lower,
            //     tree_cell_bound.upper,
            //     current_values.cell_bound.lower,
            //     current_values.cell_bound.upper,
            //
            //     intersect.lower,intersect.upper,
            //
            //     intersect.is_surface_or_volume())
            //     );

            return intersect.is_surface_or_volume();
        };
    };

    using Criterion    = InteractionCrit;
    using CriterionAcc = typename Criterion::Access;
    using CriterionVal = typename CriterionAcc::ObjectValues;

    using namespace shamrock::tree;

    q.wait();
    shambase::Timer t_tree;
    t_tree.start();
    RadixTree<u64, u64_3> tree(
        shamsys::instance::get_compute_scheduler_ptr(),
        grid.sched.get_sim_box().patch_coord_to_domain<u64_3>(p),
        pdat.get_field<u64_3>(0).get_buf(),
        pdat.get_obj_cnt(),
        0);

    tree.compute_cell_ibounding_box(q);

    tree.convert_bounding_box(q);

    q.wait();
    t_tree.end();

    TreeStructureWalker walk = generate_walk<Recompute>(
        tree.tree_struct,
        pdat.get_obj_cnt(),
        InteractionCrit{
            {},
            tree,
            pdat,
            pdat.get_field<u64_3>(0).get_buf().copy_to_sycl_buffer(),
            pdat.get_field<u64_3>(1).get_buf().copy_to_sycl_buffer()});

    sycl::buffer<u32> neighbours(len_pos);

    auto benchmark = [&]() -> f64 {
        q.wait();
        shambase::Timer t;
        t.start();

        sham::EventList depends_list;
        auto cell_min = pdat.get_field<u64_3>(0).get_buf().get_read_access(depends_list);
        auto cell_max = pdat.get_field<u64_3>(1).get_buf().get_read_access(depends_list);
        depends_list.wait_and_throw();

        q.submit([&](sycl::handler &cgh) {
             auto walker        = walk.get_access(cgh);
             auto leaf_iterator = tree.get_leaf_access(cgh);

             sycl::accessor neigh_count{neighbours, cgh, sycl::write_only, sycl::no_init};

             cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
                 u32 sum = 0;

                 CriterionVal int_values{
                     walker.criterion(), static_cast<u32>(item.get_linear_id())};

                 walker.for_each_node(
                     item,
                     int_values,
                     [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                         leaf_iterator.iter_object_in_leaf(leaf_iterator_id, [&](u32 obj_id) {
                             shammath::AABB<u64_3> cell_bound{cell_min[obj_id], cell_max[obj_id]};

                             sum += cell_bound.get_intersect(int_values.cell_bound)
                                            .is_surface_or_volume()
                                        ? 1
                                        : 0;
                         });
                     },
                     [&](u32 node_id) {});

                 neigh_count[item] = sum;
             });
         }).wait();

        t.end();
        return t.nanosec * 1e-9;
    };

    f64 time = benchmark();

    {

        f64 npart     = len_pos;
        f64 neigh_avg = 0;
        f64 neigh_var = 0;

        {
            sycl::host_accessor acc{neighbours, sycl::read_only};
            for (u32 i = 0; i < len_pos; i++) {
                neigh_avg += acc[i];
            }
            neigh_avg /= len_pos;

            for (u32 i = 0; i < len_pos; i++) {
                neigh_var += (acc[i] - neigh_avg) * (acc[i] - neigh_avg);
            }
            neigh_var /= len_pos;
        }

        Npart.push_back(npart);
        avg_neigh.push_back(neigh_avg);
        var_neigh.push_back(neigh_var);
    }

    time_walk.push_back(time);
    time_tree.push_back(t_tree.nanosec * 1e-9);
    time_refine.push_back(t_refine.nanosec * 1e-9);

    return 0;
}

template<class morton_mode, u32 reduc_lev>
void test_amr_iter_overhead_collapse(std::string dset_name) {

    std::vector<f64> Npart;
    std::vector<f64> avg_neigh;
    std::vector<f64> var_neigh;
    std::vector<f64> lambda_tilde;
    std::vector<f64> time_refine;
    std::vector<f64> time_tree;
    std::vector<f64> time_walk;

    struct FctCollapse {
        static constexpr f32 rho(f32 Rpow2) noexcept {
            constexpr f32 R_int      = 0.001;
            constexpr f32 R_int_pow2 = R_int * R_int;

            f32 div = (Rpow2 > R_int_pow2) ? Rpow2 : R_int_pow2;

            return 1 / div;
        };
    };

    for (f32 ll = 1; ll > 0.017; ll /= 1.05) {

        lambda_tilde.push_back(ll);

        amr_walk_perf<morton_mode, reduc_lev, FctCollapse>(
            ll, Npart, avg_neigh, var_neigh, time_refine, time_tree, time_walk);
    }

    auto &dat_test = shamtest::test_data().new_dataset(dset_name);

    dat_test.add_data("Ncell", Npart);
    dat_test.add_data("avg_neigh", avg_neigh);
    dat_test.add_data("var_neigh", var_neigh);
    dat_test.add_data("lambda_tilde", lambda_tilde);
    dat_test.add_data("time_refine", time_refine);
    dat_test.add_data("time_tree", time_tree);
    dat_test.add_data("time_walk", time_walk);
}

template<class morton_mode, u32 reduc_lev>
void test_amr_iter_overhead_uniform(std::string dset_name) {

    std::vector<f64> Npart;
    std::vector<f64> avg_neigh;
    std::vector<f64> var_neigh;
    std::vector<f64> lambda_tilde;
    std::vector<f64> time_refine;
    std::vector<f64> time_tree;
    std::vector<f64> time_walk;

    struct FctUniform {
        static constexpr f32 rho(f32 /*Rpow2*/) noexcept { return 1; };
    };

    for (f32 ll = 1; ll > 0.008; ll /= 1.05) {

        lambda_tilde.push_back(ll);

        amr_walk_perf<morton_mode, reduc_lev, FctUniform>(
            ll, Npart, avg_neigh, var_neigh, time_refine, time_tree, time_walk);
    }

    auto &dat_test = shamtest::test_data().new_dataset(dset_name);

    dat_test.add_data("Ncell", Npart);
    dat_test.add_data("avg_neigh", avg_neigh);
    dat_test.add_data("var_neigh", var_neigh);
    dat_test.add_data("lambda_tilde", lambda_tilde);
    dat_test.add_data("time_refine", time_refine);
    dat_test.add_data("time_tree", time_tree);
    dat_test.add_data("time_walk", time_walk);
}

template<class u_morton, class flt>
class FmmTestInteractCrit {
    using vec = sycl::vec<flt, 3>;

    public:
    RadixTree<u_morton, vec> &tree;
    sycl::buffer<vec> &positions;
    u32 leaf_count;

    RadixTreeField<flt> &cell_lengths;
    RadixTreeField<vec> &cell_centers;

    flt open_crit_sq;

    class Access {
        public:
        sycl::accessor<vec, 1, sycl::access::mode::read> part_pos;

        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_min;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_max;

        sycl::accessor<flt, 1, sycl::access::mode::read> c_length;
        sycl::accessor<vec, 1, sycl::access::mode::read> c_center;

        flt open_crit_sq;

        Access(FmmTestInteractCrit crit, sycl::handler &cgh)
            : part_pos{crit.positions, cgh, sycl::read_only},
              tree_cell_coordrange_min{
                  *crit.tree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only},
              tree_cell_coordrange_max{
                  *crit.tree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only},
              c_length{*crit.cell_lengths.radix_tree_field_buf, cgh, sycl::read_only},
              c_center{*crit.cell_centers.radix_tree_field_buf, cgh, sycl::read_only},
              open_crit_sq(crit.open_crit_sq) {}

        class ObjectValues {
            public:
            flt l_cell_a;
            vec sa;
            ObjectValues(Access acc, u32 index)
                : l_cell_a(acc.c_length[index]), sa(acc.c_center[index]) {}
        };
    };

    inline static bool criterion(
        u32 node_index, Access acc, typename Access::ObjectValues current_values) {
        vec cur_pos_min_cell_b = acc.tree_cell_coordrange_min[node_index];
        vec cur_pos_max_cell_b = acc.tree_cell_coordrange_max[node_index];

        vec sb       = acc.c_center[node_index];
        vec r_fmm    = sb - current_values.sa;
        flt l_cell_b = acc.c_length[node_index];

        flt opening_angle_sq = (current_values.l_cell_a + l_cell_b)
                               * (current_values.l_cell_a + l_cell_b) / sycl::dot(r_fmm, r_fmm);

        return (opening_angle_sq > acc.open_crit_sq);
    };
};

template<class morton_mode, class flt, u32 reduc_lev>
void test_fmm_nbody_iter_overhead(std::string dset_name, flt crit_theta) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    // setup the particle distribution

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = get_Nmax();

    u32 Nmax = 2U << 19U;

    auto coord_range = get_test_coord_ranges<vec>();

    std::vector<f64> Npart;
    std::vector<f64> Nleaf;

    std::vector<f64> avg_neigh;
    std::vector<f64> var_neigh;

    std::vector<f64> avg_excl_cells;
    std::vector<f64> var_excl_cells;

    std::vector<f64> times;

    auto mix_seed = [](f64 seed) -> u32 {
        f64 a = 16807;
        f64 m = 2147483647;
        seed  = std::fmod((a * seed), m);
        return u32_max * (seed / m);
    };

    u32 test_per_n = 10;
    u32 seed       = 0x111;
    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        for (u32 i = 0; i < 15; i++) {
            seed        = mix_seed(seed);
            u32 len_pos = cnt;

            shamlog_debug_ln(
                "TestTreePerf",
                shambase::format(
                    "dataset : {}, len={:e} seed={:10}", dset_name, f32(len_pos), seed));

            auto pos = shamalgs::random::mock_buffer_ptr<vec>(
                seed, len_pos, coord_range.lower, coord_range.upper);

            sycl::buffer<u32> neighbours(len_pos);
            sycl::buffer<u32> excl_cells(len_pos);

            RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
                shamsys::instance::get_compute_queue(),
                {coord_range.lower, coord_range.upper},
                pos,
                cnt,
                reduc_lev);

            rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            rtree.convert_bounding_box(shamsys::instance::get_compute_queue());

            RadixTreeField<flt> cell_lengths;
            RadixTreeField<vec> cell_centers;

            cell_lengths.nvar = 1;
            cell_centers.nvar = 1;

            auto &buf_cell_length  = cell_lengths.radix_tree_field_buf;
            auto &buf_cell_centers = cell_centers.radix_tree_field_buf;

            buf_cell_centers = std::make_unique<sycl::buffer<vec>>(
                rtree.tree_struct.internal_cell_count
                + rtree.tree_reduced_morton_codes.tree_leaf_count);
            buf_cell_length = std::make_unique<sycl::buffer<flt>>(
                rtree.tree_struct.internal_cell_count
                + rtree.tree_reduced_morton_codes.tree_leaf_count);

            q.submit([&](sycl::handler &cgh) {
                sycl::range<1> range_tree = sycl::range<1>{
                    rtree.tree_reduced_morton_codes.tree_leaf_count
                    + rtree.tree_struct.internal_cell_count};

                auto pos_min_cell = sycl::accessor{
                    *rtree.tree_cell_ranges.buf_pos_min_cell_flt, cgh, sycl::read_only};
                auto pos_max_cell = sycl::accessor{
                    *rtree.tree_cell_ranges.buf_pos_max_cell_flt, cgh, sycl::read_only};

                auto c_centers
                    = sycl::accessor{*buf_cell_centers, cgh, sycl::write_only, sycl::no_init};
                auto c_length
                    = sycl::accessor{*buf_cell_length, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                    vec cur_pos_min_cell_a = pos_min_cell[item];
                    vec cur_pos_max_cell_a = pos_max_cell[item];

                    vec sa = (cur_pos_min_cell_a + cur_pos_max_cell_a) / 2;

                    vec dc_a = (cur_pos_max_cell_a - cur_pos_min_cell_a);

                    flt l_cell_a = sycl::max(sycl::max(dc_a.x(), dc_a.y()), dc_a.z());

                    c_centers[item] = sa;
                    c_length[item]  = l_cell_a;
                });
            });

            auto benchmark = [&]() -> f64 {
                shambase::Timer t;

                q.wait();
                t.start();

                using Criterion    = FmmTestInteractCrit<morton_mode, flt>;
                using CriterionAcc = typename Criterion::Access;
                using CriterionVal = typename CriterionAcc::ObjectValues;

                using namespace shamrock::tree;
                TreeStructureWalker walk = generate_walk<Recompute>(
                    rtree.tree_struct,
                    rtree.tree_reduced_morton_codes.tree_leaf_count,
                    FmmTestInteractCrit<morton_mode, flt>{
                        rtree,
                        *pos,
                        rtree.tree_reduced_morton_codes.tree_leaf_count,
                        cell_lengths,
                        cell_centers,
                        crit_theta * crit_theta});

                q.submit([&](sycl::handler &cgh) {
                    auto walker        = walk.get_access(cgh);
                    auto leaf_iterator = rtree.get_leaf_access(cgh);

                    sycl::accessor neigh_count{neighbours, cgh, sycl::write_only, sycl::no_init};
                    sycl::accessor excl_count_acc{excl_cells, cgh, sycl::write_only, sycl::no_init};

                    cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
                        u32 sum_found = 0;
                        u32 sum_excl  = 0;

                        CriterionVal int_values{
                            walker.criterion(), static_cast<u32>(item.get_linear_id())};

                        walker.for_each_node(
                            item,
                            int_values,
                            [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                                leaf_iterator.iter_object_in_leaf(
                                    leaf_iterator_id, [&](u32 obj_id) {
                                        sum_found += 1;
                                    });
                            },
                            [&](u32 node_id) {
                                sum_excl += 1;
                            });

                        leaf_iterator.iter_object_in_leaf(item.get_linear_id(), [&](u32 obj_id) {
                            neigh_count[obj_id]    = sum_found;
                            excl_count_acc[obj_id] = sum_excl;
                        });
                    });
                });

                q.wait();
                t.end();

                return t.nanosec * 1e-9;
            };

            f64 time = benchmark();

            {

                f64 npart     = len_pos;
                f64 neigh_avg = 0;
                f64 neigh_var = 0;
                f64 excl_avg  = 0;
                f64 excl_var  = 0;

                {
                    sycl::host_accessor acc{neighbours, sycl::read_only};
                    sycl::host_accessor acc2{excl_cells, sycl::read_only};

                    for (u32 i = 0; i < len_pos; i++) {
                        neigh_avg += acc[i];
                        excl_avg += acc2[i];
                    }
                    neigh_avg /= len_pos;
                    excl_avg /= len_pos;

                    for (u32 i = 0; i < len_pos; i++) {
                        neigh_var += (acc[i] - neigh_avg) * (acc[i] - neigh_avg);
                        excl_var += (acc2[i] - excl_avg) * (acc2[i] - excl_avg);
                    }
                    neigh_var /= len_pos;
                    excl_var /= len_pos;
                }

                Npart.push_back(npart);
                Nleaf.push_back(rtree.tree_reduced_morton_codes.tree_leaf_count);
                avg_neigh.push_back(neigh_avg);
                var_neigh.push_back(neigh_var);
                avg_excl_cells.push_back(excl_avg);
                var_excl_cells.push_back(excl_var);
                times.push_back(time);
                // rpart_vec.push_back(rpart);
            }
        }
    }

    auto &dat_test = shamtest::test_data().new_dataset(dset_name);

    dat_test.add_data("Nobj", Npart);
    dat_test.add_data("Nleaf", Nleaf);
    dat_test.add_data("avg_neigh", avg_neigh);
    dat_test.add_data("var_neigh", var_neigh);
    dat_test.add_data("avg_excl_cells", avg_excl_cells);
    dat_test.add_data("var_excl_cells", var_excl_cells);
    dat_test.add_data("time", times);
    // dat_test.add_data("rpart", rpart_vec);
}

TestStart(
    Benchmark, "shamrock_article1:sph_walk_perf", tree_walk_sph_paper_results_tree_perf_steps, 1) {
    test_sph_iter_overhead<u32, f32, 15>("sph uniform distrib reduction level 15");
    test_sph_iter_overhead<u32, f32, 6>("sph uniform distrib reduction level 6");
    test_sph_iter_overhead<u32, f32, 3>("sph uniform distrib reduction level 3");
    test_sph_iter_overhead<u32, f32, 0>("sph uniform distrib no reduction");
}

TestStart(
    Benchmark, "shamrock_article1:amr_walk_perf", tree_walk_amr_paper_results_tree_perf_steps, 1) {
    test_amr_iter_overhead_collapse<u32, 0>("collapse distrib no reduction");
    test_amr_iter_overhead_uniform<u32, 0>("uniform distrib no reduction");
}

TestStart(
    Benchmark,
    "shamrock_article1:tree_build_perf",
    tree_building_paper_results_tree_perf_steps,
    1) {

    test_tree_build_steps<u32, f32, 0>("morton = u32, field type = f32");
    test_tree_build_steps<u64, f32, 0>("morton = u64, field type = f32");
    test_tree_build_steps<u32, f64, 0>("morton = u32, field type = f64");
    test_tree_build_steps<u64, f64, 0>("morton = u64, field type = f64");
    test_tree_build_steps<u32, u64, 0>("morton = u32, field type = u64");
    test_tree_build_steps<u64, u64, 0>("morton = u64, field type = u64");
}

TestStart(
    Benchmark, "shamrock_article1:fmm_walk_perf", tree_walk_fmm_paper_results_tree_perf_steps, 1) {

    test_fmm_nbody_iter_overhead<u32, f32, 6>(
        "fmm uniform distrib reduction level = 6 thetac = 1", 1);
    test_fmm_nbody_iter_overhead<u32, f32, 6>(
        "fmm uniform distrib reduction level = 6 thetac = 0.75", 0.75);
    test_fmm_nbody_iter_overhead<u32, f32, 6>(
        "fmm uniform distrib reduction level = 6 thetac = 0.5", 0.5);
    test_fmm_nbody_iter_overhead<u32, f32, 6>(
        "fmm uniform distrib reduction level = 6 thetac = 0.4", 0.4);
    test_fmm_nbody_iter_overhead<u32, f32, 6>(
        "fmm uniform distrib reduction level = 6 thetac = 0.3", 0.3);

    // test_fmm_nbody_iter_overhead<u32, f32, 3>("fmm uniform distrib reduction level 3 thetac =
    // 3e-1",0.3);
    test_fmm_nbody_iter_overhead<u32, f32, 0>("fmm uniform distrib no reduction thetac = 1", 1);
    test_fmm_nbody_iter_overhead<u32, f32, 0>(
        "fmm uniform distrib no reduction thetac = 0.75", 0.75);
    test_fmm_nbody_iter_overhead<u32, f32, 0>("fmm uniform distrib no reduction thetac = 0.5", 0.5);
    test_fmm_nbody_iter_overhead<u32, f32, 0>("fmm uniform distrib no reduction thetac = 0.4", 0.4);
    test_fmm_nbody_iter_overhead<u32, f32, 0>("fmm uniform distrib no reduction thetac = 0.3", 0.3);
}
