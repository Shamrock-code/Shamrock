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
 * @file AMRGrid.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "AMRCell.hpp"
#include "shamalgs/algorithm.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTreeMortonBuilder.hpp"
#include <vector>

namespace shamrock::amr {

    struct OptIndexList {
        std::optional<sycl::buffer<u32>> idx;
        u32 count;
    };

    /**
     * @brief The AMR grid only sees the grid as an integer map
     *
     * @tparam Tcoord
     * @tparam dim
     */
    template<class Tcoord, u32 dim>
    class AMRGrid {
        public:
        PatchScheduler &sched;

        using CellCoord                  = AMRBlockCoord<Tcoord, dim>;
        static constexpr u32 dimension   = dim;
        static constexpr u32 split_count = CellCoord::splts_count;

        void check_amr_main_fields() {

            bool correct_type = true;
            correct_type &= sched.pdl.check_field_type<Tcoord>(0);
            correct_type &= sched.pdl.check_field_type<Tcoord>(1);

            bool correct_names = true;
            correct_names &= sched.pdl.get_field<Tcoord>(0).name == "cell_min";
            correct_names &= sched.pdl.get_field<Tcoord>(1).name == "cell_max";

            if (!correct_type || !correct_names) {
                throw std::runtime_error(
                    "the amr module require a layout in the form :\n"
                    "    0 : cell_min : nvar=1 type : (Coordinate type)\n"
                    "    1 : cell_max : nvar=1 type : (Coordinate type)\n\n"
                    "the current layout is : \n"
                    + sched.pdl.get_description_str());
            }
        }

        explicit AMRGrid(PatchScheduler &scheduler) : sched(scheduler) { check_amr_main_fields(); }

        /**
         * @brief generate split lists for all patchdata owned by the node
         * ~~~~~{.cpp}
         *
         * auto split_lists = grid.gen_splitlists(
         *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) -> sycl::buffer<u32> {
         *          generate the buffer saying which cells should split
         *     }
         * );
         *
         * ~~~~~
         *
         * @tparam Fct
         * @param f
         * @return shambase::DistributedData<SplitList>
         */
        shambase::DistributedData<OptIndexList> gen_refinelists_native(
            std::function<void(u64, patch::Patch, patch::PatchData &, sycl::buffer<u32> &)> fct) {

            shambase::DistributedData<OptIndexList> ret;

            using namespace patch;

            u64 tot_refine = 0;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                sycl::queue &q = shamsys::instance::get_compute_queue();

                u32 obj_cnt = pdat.get_obj_cnt();

                sycl::buffer<u32> refine_flags(obj_cnt);

                // fill in the refinment flags
                fct(id_patch, cur_p, pdat, refine_flags);

                // perform stream compactions on the refinement flags
                auto [buf, len] = shamalgs::numeric::stream_compact(q, refine_flags, obj_cnt);

                shamlog_debug_ln("AMRGrid", "patch ", id_patch, "refine cell count = ", len);

                tot_refine += len;

                // add the results to the map
                ret.add_obj(id_patch, OptIndexList{std::move(buf), len});
            });

            logger::info_ln("AMRGrid", "on this process", tot_refine, "cells were refined");

            return std::move(ret);
        }

        template<class UserAcc, class Fct, class... T>
        inline shambase::DistributedData<OptIndexList> gen_refine_list(Fct &&lambd, T &&...args) {
            using namespace shamrock::patch;

            return gen_refinelists_native([&](u64 id_patch,
                                              Patch p,
                                              PatchData &pdat,
                                              sycl::buffer<u32> &refine_flags) {
                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                sham::EventList depends_list;
                sham::EventList resulting_events;

                UserAcc uacc(depends_list, id_patch, p, pdat, args...);

                auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                    sycl::accessor refine_acc{refine_flags, cgh, sycl::write_only, sycl::no_init};

                    cgh.parallel_for(sycl::range<1>(pdat.get_obj_cnt()), [=](sycl::item<1> gid) {
                        refine_acc[gid] = lambd(gid.get_linear_id(), uacc);
                    });
                });

                resulting_events.add_event(e);
                uacc.finalize(resulting_events, id_patch, p, pdat, args...);
            });
        }

        inline u64 get_process_refine_count(shambase::DistributedData<OptIndexList> &splits) {
            u64 acc = 0;

            splits.for_each([&acc](u64 id, OptIndexList &idx_list) {
                acc += idx_list.count;
            });

            return acc;
        }

        template<class UserAcc, class Fct>
        shambase::DistributedData<OptIndexList> gen_merge_list(Fct &&lambd) {

            shambase::DistributedData<OptIndexList> ret;
            u64 tot_merge = 0;

            using MortonBuilder = RadixTreeMortonBuilder<u64, Tcoord, 3>;
            using namespace shamrock::patch;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                // return because no cell can be merged since
                if (pdat.get_obj_cnt() < split_count) {
                    return;
                }

                std::unique_ptr<sycl::buffer<u64>> out_buf_morton;
                std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;

                MortonBuilder::build(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    sched.get_sim_box().template patch_coord_to_domain<Tcoord>(cur_p),
                    pdat.get_field<Tcoord>(0).get_buf(),
                    pdat.get_obj_cnt(),
                    out_buf_morton,
                    out_buf_particle_index_map);

                // apply list permut on patch

                u32 pre_merge_obj_cnt = pdat.get_obj_cnt();

                pdat.index_remap(*out_buf_particle_index_map, pre_merge_obj_cnt);

                u32 obj_to_check = pre_merge_obj_cnt - split_count + 1;

                shamlog_debug_sycl_ln("AMR Grid", "checking mergeable in", obj_to_check, "cells");

                sycl::buffer<u32> mergeable_indexes(obj_to_check);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                sham::EventList depends_list;
                auto acc_min = pdat.get_field<Tcoord>(0).get_buf().get_write_access(depends_list);
                auto acc_max = pdat.get_field<Tcoord>(1).get_buf().get_write_access(depends_list);

                auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                    sycl::accessor acc_mergeable{
                        mergeable_indexes, cgh, sycl::write_only, sycl::no_init};

                    sycl::range<1> rnge{obj_to_check};

                    cgh.parallel_for(rnge, [=](sycl::item<1> gid) {
                        u32 id = gid.get_linear_id();

                        std::array<CellCoord, split_count> cells;

                        for (u32 lid = 0; lid < split_count; lid++) {
                            cells[lid] = CellCoord{acc_min[gid + lid], acc_max[gid + lid]};
                        }

                        acc_mergeable[gid] = CellCoord::are_mergeable(cells);
                    });
                });

                pdat.get_field<Tcoord>(0).get_buf().complete_event_state(e);
                pdat.get_field<Tcoord>(1).get_buf().complete_event_state(e);

                {
                    sham::EventList depends_list;
                    sham::EventList resulting_events;
                    UserAcc uacc(depends_list, id_patch, cur_p, pdat);

                    auto e2 = q.submit(depends_list, [&](sycl::handler &cgh) {
                        sycl::accessor acc_mergeable{mergeable_indexes, cgh, sycl::read_write};

                        cgh.parallel_for(
                            sycl::range<1>(pdat.get_obj_cnt()), [=](sycl::item<1> gid) {
                                if (acc_mergeable[gid]) {
                                    acc_mergeable[gid] = lambd(gid.get_linear_id(), uacc);
                                }
                            });
                    });

                    resulting_events.add_event(e2);
                    uacc.finalize(resulting_events, id_patch, cur_p, pdat);
                }

                auto [opt_buf, len] = shamalgs::numeric::stream_compact(
                    shamsys::instance::get_compute_queue(), mergeable_indexes, obj_to_check);

                shamlog_debug_ln("AMRGrid", "patch ", id_patch, "merge cell count = ", len);

                tot_merge += len;

                // add the results to the map
                ret.add_obj(id_patch, OptIndexList{std::move(opt_buf), len});
            });

            logger::info_ln(
                "AMRGrid", "on this process", tot_merge * split_count, "cells were derefined");

            return std::move(ret);
        }

        /**
         * @brief
         *
         * @tparam UserAcc
         * @tparam Fct
         * @param splts
         * @param lambd
         */
        template<class UserAcc, class Fct>
        void apply_splits(shambase::DistributedData<OptIndexList> &&splts, Fct &&lambd) {

            using namespace patch;

            u64 sum_cell_count = 0;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                sycl::queue &q = shamsys::instance::get_compute_queue();

                u32 old_obj_cnt = pdat.get_obj_cnt();

                OptIndexList &refine_flags = splts.get(id_patch);

                if (refine_flags.count > 0) {

                    pdat.expand(refine_flags.count * (split_count - 1));

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    sham::EventList depends_list;

                    auto cell_bound_low
                        = pdat.get_field<Tcoord>(0).get_buf().get_write_access(depends_list);
                    auto cell_bound_high
                        = pdat.get_field<Tcoord>(1).get_buf().get_write_access(depends_list);

                    sham::EventList resulting_events;
                    UserAcc uacc(depends_list, pdat);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        sycl::accessor index_to_ref{*refine_flags.idx, cgh, sycl::read_only};

                        u32 start_index_push = old_obj_cnt;

                        constexpr u32 new_splits = split_count - 1;

                        cgh.parallel_for(
                            sycl::range<1>(refine_flags.count), [=](sycl::item<1> gid) {
                                u32 tid = gid.get_linear_id();

                                u32 idx_to_refine = index_to_ref[gid];

                                // gen splits coordinates
                                CellCoord cur_cell{
                                    cell_bound_low[idx_to_refine], cell_bound_high[idx_to_refine]};

                                std::array<CellCoord, split_count> cell_coords
                                    = CellCoord::get_split(cur_cell.bmin, cur_cell.bmax);

                                // generate index for the refined cells
                                std::array<u32, split_count> cells_ids;
                                cells_ids[0] = idx_to_refine;

#pragma unroll
                                for (u32 pid = 0; pid < new_splits; pid++) {
                                    cells_ids[pid + 1] = start_index_push + tid * new_splits + pid;
                                }

                            // write coordinates

#pragma unroll
                                for (u32 pid = 0; pid < split_count; pid++) {
                                    cell_bound_low[cells_ids[pid]]  = cell_coords[pid].bmin;
                                    cell_bound_high[cells_ids[pid]] = cell_coords[pid].bmax;
                                }

                                // user lambda to fill the fields
                                lambd(idx_to_refine, cur_cell, cells_ids, cell_coords, uacc);
                            });
                    });

                    pdat.get_field<Tcoord>(0).get_buf().complete_event_state(e);
                    pdat.get_field<Tcoord>(1).get_buf().complete_event_state(e);

                    resulting_events.add_event(e);
                    uacc.finalize(resulting_events, pdat);
                }

                sum_cell_count += pdat.get_obj_cnt();
            });

            logger::info_ln("AMRGrid", "process cell count =", sum_cell_count);
        }

        template<class UserAcc, class Fct>
        void apply_merge(shambase::DistributedData<OptIndexList> &&splts, Fct &&lambd) {

            using namespace patch;

            sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                u32 old_obj_cnt = pdat.get_obj_cnt();

                OptIndexList &derefine_flags = splts.get(id_patch);

                if (derefine_flags.count > 0) {

                    // init flag table
                    sycl::buffer<u32> keep_cell_flag = shamalgs::algorithm::gen_buffer_device(
                        q.q, old_obj_cnt, [](u32 i) -> u32 {
                            return 1;
                        });

                    sham::EventList depends_list;
                    auto cell_bound_low
                        = pdat.get_field<Tcoord>(0).get_buf().get_write_access(depends_list);
                    auto cell_bound_high
                        = pdat.get_field<Tcoord>(1).get_buf().get_write_access(depends_list);

                    sham::EventList resulting_events;
                    UserAcc uacc(depends_list, pdat);

                    // edit cell content + make flag of cells to keep
                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        sycl::accessor index_to_deref{*derefine_flags.idx, cgh, sycl::read_only};

                        sycl::accessor flag_keep{keep_cell_flag, cgh, sycl::read_write};

                        cgh.parallel_for(
                            sycl::range<1>(derefine_flags.count), [=](sycl::item<1> gid) {
                                u32 tid = gid.get_linear_id();

                                u32 idx_to_derefine = index_to_deref[gid];

                                // compute old cell indexes
                                std::array<u32, split_count> old_indexes;
#pragma unroll
                                for (u32 pid = 0; pid < split_count; pid++) {
                                    old_indexes[pid] = idx_to_derefine + pid;
                                }

                                // load cell coords
                                std::array<CellCoord, split_count> cell_coords;
#pragma unroll
                                for (u32 pid = 0; pid < split_count; pid++) {
                                    cell_coords[pid] = CellCoord{
                                        cell_bound_low[old_indexes[pid]],
                                        cell_bound_high[old_indexes[pid]]};
                                }

                                // make new cell coord
                                CellCoord merged_cell_coord = CellCoord::get_merge(cell_coords);

                                // write new coord
                                cell_bound_low[idx_to_derefine]  = merged_cell_coord.bmin;
                                cell_bound_high[idx_to_derefine] = merged_cell_coord.bmax;

// flag the old cells for removal
#pragma unroll
                                for (u32 pid = 1; pid < split_count; pid++) {
                                    flag_keep[idx_to_derefine + pid] = 0;
                                }

                                // user lambda to fill the fields
                                lambd(
                                    old_indexes,
                                    cell_coords,
                                    idx_to_derefine,
                                    merged_cell_coord,
                                    uacc);
                            });
                    });

                    pdat.get_field<Tcoord>(0).get_buf().complete_event_state(e);
                    pdat.get_field<Tcoord>(1).get_buf().complete_event_state(e);
                    resulting_events.add_event(e);
                    uacc.finalize(resulting_events, pdat);

                    // stream compact the flags
                    auto [opt_buf, len]
                        = shamalgs::numeric::stream_compact(q.q, keep_cell_flag, old_obj_cnt);

                    shamlog_debug_ln(
                        "AMR Grid",
                        "patch",
                        id_patch,
                        "derefine cell count ",
                        old_obj_cnt,
                        "->",
                        len);

                    if (!opt_buf) {
                        throw std::runtime_error("opt buf must contain something at this point");
                    }

                    // remap pdat according to stream compact
                    pdat.index_remap_resize(*opt_buf, len);
                }
            });
        }

        inline void make_base_grid(Tcoord bmin, Tcoord cell_size, std::array<u32, dim> cell_count) {

            Tcoord bmax{
                bmin.x() + cell_size.x() * (cell_count[0]),
                bmin.y() + cell_size.y() * (cell_count[1]),
                bmin.z() + cell_size.z() * (cell_count[2])};

            sched.set_coord_domain_bound(bmin, bmax);

            if ((cell_size.x() != cell_size.y()) || (cell_size.y() != cell_size.z())) {
                logger::warn_ln("AMR Grid", "your cells aren't cube");
            }

            static_assert(dim == 3, "this is not implemented for dim != 3");

            std::array<u32, dim> patch_count;

            constexpr u32 gcd_pow2 = 1U << 31U;
            u32 gcd_cell_count;
            {
                gcd_cell_count = std::gcd(cell_count[0], cell_count[1]);
                gcd_cell_count = std::gcd(gcd_cell_count, cell_count[2]);
                gcd_cell_count = std::gcd(gcd_cell_count, gcd_pow2);
            }

            shamlog_debug_ln(
                "AMRGrid",
                "patch grid :",
                cell_count[0] / gcd_cell_count,
                cell_count[1] / gcd_cell_count,
                cell_count[2] / gcd_cell_count);

            sched.make_patch_base_grid<3>(
                {{cell_count[0] / gcd_cell_count,
                  cell_count[1] / gcd_cell_count,
                  cell_count[2] / gcd_cell_count}});

            sched.for_each_patch([](u64 id_patch, patch::Patch p) {
                // TODO implement check to verify that patch a cubes of size 2^n
            });

            u32 cell_tot_count = cell_count[0] * cell_count[1] * cell_count[2];

            sycl::buffer<Tcoord> cell_coord_min(cell_tot_count);
            sycl::buffer<Tcoord> cell_coord_max(cell_tot_count);

            shamlog_debug_sycl_ln(
                "AMRGrid", "building bounds ", cell_count[0], cell_count[1], cell_count[2]);

            {
                sycl::host_accessor acc_min{cell_coord_min, sycl::write_only, sycl::no_init};
                sycl::host_accessor acc_max{cell_coord_max, sycl::write_only, sycl::no_init};

                sycl::range<3> rnge{cell_count[0], cell_count[1], cell_count[2]};

                u32 cnt_x = cell_count[0];
                u32 cnt_y = cell_count[1];
                u32 cnt_z = cell_count[2];

                u32 cnt_xy = cnt_x * cnt_y;

                Tcoord sz = cell_size;

                for (u64 idx = 0; idx < cell_count[0]; idx++) {
                    for (u64 idy = 0; idy < cell_count[1]; idy++) {
                        for (u64 idz = 0; idz < cell_count[2]; idz++) {

                            u64 id_a = idx + cnt_x * idy + cnt_xy * idz;

                            acc_min[id_a] = sz * Tcoord{idx, idy, idz};
                            acc_max[id_a] = sz * Tcoord{idx + 1, idy + 1, idz + 1};
                        }
                    }
                }
            }

            shambase::check_queue_state(shamsys::instance::get_compute_queue());

            patch::PatchData pdat(sched.pdl);
            pdat.resize(cell_tot_count);

            shambase::check_queue_state(shamsys::instance::get_compute_queue());
            pdat.get_field<Tcoord>(0).override(cell_coord_min, cell_tot_count);

            shambase::check_queue_state(shamsys::instance::get_compute_queue());
            pdat.get_field<Tcoord>(1).override(cell_coord_max, cell_tot_count);

            shambase::check_queue_state(shamsys::instance::get_compute_queue());

            sched.allpush_data(pdat);

            shambase::check_queue_state(shamsys::instance::get_compute_queue());

            shamlog_debug_sycl_ln("AMRGrid", "grid init done");
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // template<class Tcoord, u32 dim>
    // inline auto
    // AMRGrid<Tcoord, dim>::gen_splitlists(std::function<sycl::buffer<u32>(u64 , patch::Patch ,
    // patch::PatchData &)> fct) -> shambase::DistributedData<SplitList> {
    //
    //    shambase::DistributedData<SplitList> ret;
    //
    //    using namespace patch;
    //
    //    sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
    //        sycl::queue &q = shamsys::instance::get_compute_queue();
    //
    //        u32 obj_cnt = pdat.get_obj_cnt();
    //
    //        sycl::buffer<u32> split_flags = fct(id_patch, cur_p, pdat);
    //
    //        auto [buf, len] = shamalgs::numeric::stream_compact(q, split_flags, obj_cnt);
    //
    //        ret.add_obj(id_patch, SplitList{std::move(buf), len});
    //    });
    //
    //    return std::move(ret);
    //}

} // namespace shamrock::amr
