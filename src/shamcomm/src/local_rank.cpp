// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file local_rank.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Functions related to the MPI communicator
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamcomm/env_variables.hpp"
#include <optional>

namespace shamcomm {
    std::optional<u32> node_local_rank() {

        if (MV2_COMM_WORLD_LOCAL_RANK) {
            return std::atoi(MV2_COMM_WORLD_LOCAL_RANK->c_str());
        }

        if (OMPI_COMM_WORLD_LOCAL_RANK) {
            return std::atoi(OMPI_COMM_WORLD_LOCAL_RANK->c_str());
        }

        if (MPI_LOCALRANKID) {
            return std::atoi(MPI_LOCALRANKID->c_str());
        }

        if (SLURM_PROCID) {
            return std::atoi(SLURM_PROCID->c_str());
        }

        if (LOCAL_RANK) {
            return std::atoi(LOCAL_RANK->c_str());
        }

        if (PALS_LOCAL_RANKID) {
            return std::atoi(PALS_LOCAL_RANKID->c_str());
        }

        return {};
    }

    bool is_main_node_rank(u32 main_local_rank_id) {
        auto loc_r = node_local_rank();
        return (loc_r) ? *loc_r == main_local_rank_id : true;
    }

} // namespace shamcomm
