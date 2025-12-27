// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file VTKDumpUtils.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Shared VTK dump utilities for SPH-based models
 *
 * Contains common helper functions for VTK output that are shared
 * between SPH and GSPH models.
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/memory.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::io {

    /**
     * @brief Start a VTK dump by writing particle positions
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @param sched Patch scheduler
     * @param dump_name Output filename
     * @return VTK writer ready for additional fields
     */
    template<class Tvec>
    inline shamrock::LegacyVtkWritter start_dump(
        PatchScheduler &sched, const std::string &dump_name) {
        StackEntry stack_loc{};
        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = sched.get_rank_count();

        shamlog_debug_mpi_ln("VTKDump", "rank count =", num_obj);

        std::unique_ptr<sycl::buffer<Tvec>> pos = sched.rankgather_field<Tvec>(0);

        writer.write_points(pos, num_obj);

        return writer;
    }

    /**
     * @brief Add patch ID field to VTK dump
     *
     * @param sched Patch scheduler
     * @param writer VTK writer
     */
    inline void vtk_dump_add_patch_id(PatchScheduler &sched, shamrock::LegacyVtkWritter &writer) {
        StackEntry stack_loc{};

        u64 num_obj = sched.get_rank_count();

        using namespace shamrock::patch;

        if (num_obj > 0) {
            sycl::buffer<u64> idp(num_obj);

            u64 ptr = 0;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                using namespace shamalgs::memory;
                using namespace shambase;

                write_with_offset_into(
                    shamsys::instance::get_compute_queue(),
                    idp,
                    cur_p.id_patch,
                    ptr,
                    pdat.get_obj_cnt());

                ptr += pdat.get_obj_cnt();
            });

            writer.write_field("patchid", idp, num_obj);
        } else {
            writer.write_field_no_buf<u64>("patchid");
        }
    }

    /**
     * @brief Add world rank field to VTK dump
     *
     * @param sched Patch scheduler
     * @param writer VTK writer
     */
    inline void vtk_dump_add_worldrank(PatchScheduler &sched, shamrock::LegacyVtkWritter &writer) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        if (num_obj > 0) {
            sycl::buffer<u32> idp(num_obj);

            u64 ptr = 0;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                using namespace shamalgs::memory;
                using namespace shambase;

                write_with_offset_into<u32>(
                    shamsys::instance::get_compute_queue(),
                    idp,
                    shamcomm::world_rank(),
                    ptr,
                    pdat.get_obj_cnt());

                ptr += pdat.get_obj_cnt();
            });

            writer.write_field("world_rank", idp, num_obj);
        } else {
            writer.write_field_no_buf<u32>("world_rank");
        }
    }

    /**
     * @brief Add a compute field to VTK dump
     *
     * @tparam T Field type (scalar or vector)
     * @param sched Patch scheduler
     * @param writer VTK writer
     * @param field Compute field to write
     * @param field_dump_name Name of the field in VTK output
     */
    template<class T>
    inline void vtk_dump_add_compute_field(
        PatchScheduler &sched,
        shamrock::LegacyVtkWritter &writer,
        shamrock::ComputeField<T> &field,
        const std::string &field_dump_name) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        if (num_obj > 0) {
            std::unique_ptr<sycl::buffer<T>> field_vals = field.rankgather_computefield(sched);

            writer.write_field(field_dump_name, field_vals, num_obj);
        } else {
            writer.write_field_no_buf<T>(field_dump_name);
        }
    }

    /**
     * @brief Add a data field to VTK dump
     *
     * @tparam T Field type (scalar or vector)
     * @param sched Patch scheduler
     * @param writer VTK writer
     * @param field_idx Field index in patch data
     * @param field_dump_name Name of the field in VTK output
     */
    template<class T>
    inline void vtk_dump_add_field(
        PatchScheduler &sched,
        shamrock::LegacyVtkWritter &writer,
        u32 field_idx,
        const std::string &field_dump_name) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        u64 num_obj = sched.get_rank_count();

        if (num_obj > 0) {
            std::unique_ptr<sycl::buffer<T>> field_vals = sched.rankgather_field<T>(field_idx);

            writer.write_field(field_dump_name, field_vals, num_obj);
        } else {
            writer.write_field_no_buf<T>(field_dump_name);
        }
    }

} // namespace shammodels::common::io
