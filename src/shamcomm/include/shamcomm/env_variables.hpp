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
 * @file env_variables.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamcmdopt/env.hpp"
#include <cstdlib>

namespace shamcomm {
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

    const std::optional<std::string> PSM2_CUDA = shamcmdopt::getenv_str("PSM2_CUDA");
} // namespace shamcomm
