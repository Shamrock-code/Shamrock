// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file worldInfo.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Functions related to the MPI communicator
 *
 * This header provides functions to get information about the MPI communicator.
 */

#include "shambase/aliases_int.hpp"
#include <string>

namespace shamcomm {

    /**
     * @brief Gives the rank of the current process in the MPI communicator
     *
     * @return the rank of the current process in the MPI communicator
     */
    i32 world_rank();

    /**
     * @brief Gives the size of the MPI communicator
     *
     * @return the size of the MPI communicator
     */
    i32 world_size();

    /**
     * @brief Gets the maximum value of the MPI tag
     *
     * @return the maximum value of the MPI tag
     */
    i32 mpi_max_tag_value();

    /**
     * @brief Gets the name of the MPI process
     *
     * @return the name of the MPI process
     */
    std::string mpi_process_name();

    /**
     * @brief Gets the information about the MPI communicator
     *
     * This function queries the MPI communicator size and rank. It should be called once in the
     * program.
     */
    void fetch_world_info();

    /**
     * @brief Check if MPI is initialized
     *
     * This function returns true if MPI has been initialized, false otherwise.
     *
     * @return true if MPI is initialized, false otherwise
     */
    bool is_mpi_initialized();

} // namespace shamcomm

/// @brief Macro to execute code only on rank 0
#define ON_RANK_0(x)                                                                               \
    do {                                                                                           \
        if (shamcomm::world_rank() == 0) {                                                         \
            x;                                                                                     \
        }                                                                                          \
    } while (false)
