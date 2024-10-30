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
 * @file worldInfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Functions related to the MPI communicator
 *
 * This header provides functions to get information about the MPI communicator.
 */

#include "shambase/aliases_int.hpp"

namespace shamcomm {

    /**
     * @brief Gives the rank of the current process in the MPI communicator
     *
     * @return the rank of the current process in the MPI communicator
     */
    const i32 world_rank();

    /**
     * @brief Gives the size of the MPI communicator
     *
     * @return the size of the MPI communicator
     */
    const i32 world_size();

    /**
     * @brief Gets the information about the MPI communicator
     *
     * This function queries the MPI communicator size and rank. It should be called once in the
     * program.
     */
    void fetch_world_info();

} // namespace shamcomm
