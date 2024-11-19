// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMROverheadtest.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/time.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/intervals.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeStructureWalker.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <vector>

class AMRTestModel {
    public:
    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;
    Grid &grid;

    explicit AMRTestModel(Grid &grd) : grid(grd) {}

    class RefineCritCellAccessor {
        public:
        const u64_3 *cell_low_bound;
        const u64_3 *cell_high_bound;

        RefineCritCellAccessor(
            sycl::handler &cgh,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchData &pdat)
            : cell_low_bound{pdat.get_field<u64_3>(0).get_buf().get_read_access_sync()},
              cell_high_bound{pdat.get_field<u64_3>(1).get_buf().get_read_access_sync()} {}
    };

    template<class T>
    using buf_access_read = sycl::accessor<T, 1, sycl::access::mode::read, sycl::target::device>;
    template<class T>
    using buf_access_read_write
        = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::target::device>;

    class RefineCellAccessor {
        public:
        u32 *field;

        RefineCellAccessor(sycl::handler &cgh, shamrock::patch::PatchData &pdat)
            : field{pdat.get_field<u32>(2).get_buf().get_write_access_sync()} {}
    };

    inline void dump_patch(u64 id) {

        using namespace shamrock::patch;
        using namespace shamalgs::memory;

        PatchData &pdat = grid.sched.patch_data.get_pdat(id);

        std::vector<u64_3> mins = pdat.get_field<u64_3>(0).get_buf().copy_to_stdvec();
        std::vector<u64_3> maxs = pdat.get_field<u64_3>(1).get_buf().copy_to_stdvec();

        logger::raw_ln("----- dump");
        for (u32 i = 0; i < mins.size(); i++) {
            logger::raw_ln(mins[i], maxs[i]);
        }
        logger::raw_ln("-----");
    }

    static constexpr u64 fact_p_len = 2;

    /**
     * @brief does the refinment step of the AMR
     *
     */
    inline void refine() {

        // dump_patch(4);
        auto splits = grid.gen_refine_list<RefineCritCellAccessor>(
            [](u32 cell_id, RefineCritCellAccessor acc) -> u32 {
                u64_3 low_bound  = acc.cell_low_bound[cell_id];
                u64_3 high_bound = acc.cell_high_bound[cell_id];

                using namespace shammath;

                bool should_refine
                    = is_in_half_open(
                          low_bound, fact_p_len * u64_3{1, 1, 1}, fact_p_len * u64_3{4, 4, 4})
                      && is_in_half_open(
                          high_bound, fact_p_len * u64_3{1, 1, 1}, fact_p_len * u64_3{4, 4, 4});

                should_refine = should_refine && (high_bound.x() - low_bound.x() > 1);
                should_refine = should_refine && (high_bound.y() - low_bound.y() > 1);
                should_refine = should_refine && (high_bound.z() - low_bound.z() > 1);

                return should_refine;
            });

        grid.apply_splits<RefineCellAccessor>(
            std::move(splits),

            [](u32 cur_idx,
               Grid::CellCoord cur_coords,
               std::array<u32, 8> new_cells,
               std::array<Grid::CellCoord, 8> new_cells_coords,
               RefineCellAccessor acc) {
                u32 val = acc.field[cur_idx];

#pragma unroll
                for (u32 pid = 0; pid < 8; pid++) {
                    acc.field[new_cells[pid]] = val;
                }
            }

        );

        // dump_patch(4);
    }

    inline void derefine() {
        auto merge = grid.gen_merge_list<RefineCritCellAccessor>(
            [](u32 cell_id, RefineCritCellAccessor acc) -> u32 {
                u64_3 low_bound  = acc.cell_low_bound[cell_id];
                u64_3 high_bound = acc.cell_high_bound[cell_id];

                using namespace shammath;

                bool should_merge
                    = is_in_half_open(
                          low_bound, fact_p_len * u64_3{1, 1, 1}, fact_p_len * u64_3{4, 4, 4})
                      && is_in_half_open(
                          high_bound, fact_p_len * u64_3{1, 1, 1}, fact_p_len * u64_3{4, 4, 4});

                return should_merge;
            });

        grid.apply_merge<RefineCellAccessor>(
            std::move(merge),

            [](std::array<u32, 8> old_cells,
               std::array<Grid::CellCoord, 8> old_coords,
               u32 new_cell,
               Grid::CellCoord new_coord,

               RefineCellAccessor acc) {
                u32 accum = 0;

#pragma unroll
                for (u32 pid = 0; pid < 8; pid++) {
                    accum += acc.field[old_cells[pid]];
                }

                acc.field[new_cell] = accum / 8;
            }

        );
        // dump_patch(4);
    }

    inline void step() {

        using namespace shamrock::patch;

        refine();
        derefine();

        using namespace shamrock::patch;

        sycl::queue &q = shamsys::instance::get_compute_queue();

        grid.sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            RadixTree<u64, u64_3> tree(
                shamsys::instance::get_compute_scheduler_ptr(),
                grid.sched.get_sim_box().patch_coord_to_domain<u64_3>(cur_p),
                pdat.get_field<u64_3>(0).get_buf(),
                pdat.get_obj_cnt(),
                0);

            tree.compute_cell_ibounding_box(q);

            tree.convert_bounding_box(q);

            class WalkAccessors {
                public:
                u32 *field;

                WalkAccessors(sycl::handler &cgh, shamrock::patch::PatchData &pdat)
                    : field{pdat.get_field<u32>(2).get_buf().get_write_access_sync()} {}
            };

            q.wait();

            shambase::Timer t;
            t.start();

            q.submit([&](sycl::handler &cgh) {
                using Rta = walker::Radix_tree_accessor<u64, u64_3>;
                Rta tree_acc(tree, cgh);

                WalkAccessors uacc(cgh, pdat);

                sycl::range range_npart{pdat.get_obj_cnt()};

                auto cell_low_bound  = pdat.get_field<u64_3>(0).get_buf().get_read_access_sync();
                auto cell_high_bound = pdat.get_field<u64_3>(1).get_buf().get_read_access_sync();

                cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                    u64_3 low_bound_a  = cell_low_bound[item];
                    u64_3 high_bound_a = cell_high_bound[item];

                    u32 sum = 0;

                    walker::rtree_for(
                        tree_acc,
                        [&](u32 node_id) {
                            u64_3 cur_pos_min_cell_b = tree_acc.pos_min_cell[node_id];
                            u64_3 cur_pos_max_cell_b = tree_acc.pos_max_cell[node_id];

                            return shammath::domain_are_connected(
                                low_bound_a, high_bound_a, cur_pos_min_cell_b, cur_pos_max_cell_b);
                        },
                        [&](u32 id_b) {
                            // compute only omega_a

                            sum += 1;
                        },
                        [](u32 node_id) {});

                    uacc.field[item] = sum;
                });
            });
            q.wait();
            t.end();

            logger::debug_ln("AMR Test", "walk time", t.get_time_str());

            class InteractionCrit {
                public:
                shammath::CoordRange<u64_3> bounds;

                RadixTree<u64, u64_3> &tree;
                PatchData &pdat;

                class Access {
                    public:
                    const u64_3 *cell_low_bound;
                    const u64_3 *cell_high_bound;

                    sycl::accessor<u64_3, 1, sycl::access::mode::read> tree_cell_coordrange_min;
                    sycl::accessor<u64_3, 1, sycl::access::mode::read> tree_cell_coordrange_max;

                    Access(InteractionCrit crit, sycl::handler &cgh)
                        : cell_low_bound{crit.pdat.get_field<u64_3>(0)
                                             .get_buf()
                                             .get_read_access_sync()},
                          cell_high_bound{
                              crit.pdat.get_field<u64_3>(1).get_buf().get_read_access_sync()},
                          tree_cell_coordrange_min{
                              *crit.tree.tree_cell_ranges.buf_pos_min_cell_flt,
                              cgh,
                              sycl::read_only},
                          tree_cell_coordrange_max{
                              *crit.tree.tree_cell_ranges.buf_pos_max_cell_flt,
                              cgh,
                              sycl::read_only} {}

                    class ObjectValues {
                        public:
                        u64_3 cell_low_bound;
                        u64_3 cell_high_bound;
                        ObjectValues(Access acc, u32 index)
                            : cell_low_bound(acc.cell_low_bound[index]),
                              cell_high_bound(acc.cell_high_bound[index]) {}
                    };
                };

                static bool
                criterion(u32 node_index, Access acc, Access::ObjectValues current_values) {
                    u64_3 cur_pos_min_cell_b = acc.tree_cell_coordrange_min[node_index];
                    u64_3 cur_pos_max_cell_b = acc.tree_cell_coordrange_max[node_index];

                    return shammath::domain_are_connected(
                        current_values.cell_low_bound,
                        current_values.cell_high_bound,
                        cur_pos_min_cell_b,
                        cur_pos_max_cell_b);
                };
            };

            using Criterion    = InteractionCrit;
            using CriterionAcc = typename Criterion::Access;
            using CriterionVal = typename CriterionAcc::ObjectValues;

            using namespace shamrock::tree;

            TreeStructureWalker walk = generate_walk<Recompute>(
                tree.tree_struct, pdat.get_obj_cnt(), InteractionCrit{{}, tree, pdat});

            q.submit([&](sycl::handler &cgh) {
                auto walker        = walk.get_access(cgh);
                auto leaf_iterator = tree.get_leaf_access(cgh);

                cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
                    u32 sum = 0;

                    CriterionVal int_values{
                        walker.criterion(), static_cast<u32>(item.get_linear_id())};

                    walker.for_each_node(
                        item,
                        int_values,
                        [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                            leaf_iterator.iter_object_in_leaf(
                                leaf_iterator_id, [&](u32 /*obj_id*/) {
                                    sum += 1;
                                });
                        },
                        [&](u32 node_id) {});
                });
            });
        });
    }
};
