// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file worldInfo.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/numeric_limits.hpp"
#include "shambase/profiling/chrome.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcomm/env_variables.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <optional>

namespace shamcomm {

    i32 _world_rank;

    i32 _world_size;

    std::optional<i32> _max_tag = std::nullopt;

    std::string _mpi_process_name;

    i32 mpi_max_tag_value() { return (_max_tag) ? *_max_tag : shambase::get_max<i32>(); }

    i32 world_size() { return _world_size; }

    i32 world_rank() { return _world_rank; }

    std::string mpi_process_name() { return _mpi_process_name; }

    __attribute__((no_sanitize_address)) void fetch_mpi_process_name() {
        char hn[MPI_MAX_PROCESSOR_NAME];
        int hn_len;

        MPICHECK(MPI_Get_processor_name(hn, &hn_len));

        _mpi_process_name = std::string(hn, size_t(hn_len));
    }

    void set_callstack_process_identifier() {
        std::optional<u32> local_rank = get_local_rank();

        shambase::set_callstack_process_identifier(
            shambase::format(
                "{} (world rank: {}, local rank: {})",
                _mpi_process_name,
                world_rank(),
                (local_rank) ? std::to_string(*local_rank) : "Unknown"));
    }

    void fetch_world_info() {

        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &_world_size));
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank));

        fetch_mpi_process_name();
        set_callstack_process_identifier();

        {
            void *max_tag;
            int flag;
            /* The address of a void pointer must be used! */
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
            if (flag) {
                _max_tag = *(int *) max_tag;
            }
        }

        shamcomm::mpi::Barrier(MPI_COMM_WORLD);

        shambase::profiling::chrome::set_time_offset(shambase::details::get_wtime());

        shambase::profiling::chrome::set_chrome_pid(world_rank());
    }

    bool is_mpi_initialized() {
        int flag = false;
        MPICHECK(MPI_Initialized(&flag));
        return flag;
    }

} // namespace shamcomm
