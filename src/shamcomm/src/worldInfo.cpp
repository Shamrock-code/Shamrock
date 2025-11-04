// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <optional>

namespace shamcomm {

    i32 _world_rank;

    i32 _world_size;

    std::optional<i32> _max_tag = std::nullopt;

    i32 mpi_max_tag_value() { return (_max_tag) ? *_max_tag : shambase::get_max<i32>(); }

    i32 world_size() { return _world_size; }

    i32 world_rank() { return _world_rank; }

    void fetch_world_info() {

        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &_world_size));
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank));

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
