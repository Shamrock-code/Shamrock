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
 * @file EnvVariables.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamcmdopt/env.hpp"
#include "shamsys/legacy/log.hpp"
#include <cstdlib>
#include <optional>
#include <string>

namespace shamsys::env {

    // rank related env variable
    const std::optional<std::string> MV2_COMM_WORLD_LOCAL_RANK
        = shamcmdopt::getenv_str("MV2_COMM_WORLD_LOCAL_RANK");
    const std::optional<std::string> OMPI_COMM_WORLD_LOCAL_RANK
        = shamcmdopt::getenv_str("OMPI_COMM_WORLD_LOCAL_RANK");
    const std::optional<std::string> MPI_LOCALRANKID = shamcmdopt::getenv_str("MPI_LOCALRANKID");
    const std::optional<std::string> SLURM_PROCID    = shamcmdopt::getenv_str("SLURM_PROCID");
    const std::optional<std::string> LOCAL_RANK      = shamcmdopt::getenv_str("LOCAL_RANK");
    const std::optional<std::string> PALS_LOCAL_RANKID
        = shamcmdopt::getenv_str("PALS_LOCAL_RANKID");

    inline std::optional<u32> get_local_rank() {

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

    const std::optional<std::string> PSM2_CUDA = shamcmdopt::getenv_str("PSM2_CUDA");

} // namespace shamsys::env
