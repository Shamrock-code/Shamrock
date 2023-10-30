// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EnvVariables.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamsys/legacy/log.hpp"
#include <cstdlib>
#include <optional>
#include <string>

namespace shamsys::env {

    std::optional<std::string> getenv_str(const char *env_var);

    // rank related env variable
    const std::optional<std::string> MV2_COMM_WORLD_LOCAL_RANK =
        getenv_str("MV2_COMM_WORLD_LOCAL_RANK");
    const std::optional<std::string> OMPI_COMM_WORLD_LOCAL_RANK =
        getenv_str("OMPI_COMM_WORLD_LOCAL_RANK");
    const std::optional<std::string> MPI_LOCALRANKID = getenv_str("MPI_LOCALRANKID");
    const std::optional<std::string> SLURM_PROCID    = getenv_str("SLURM_PROCID");
    const std::optional<std::string> LOCAL_RANK      = getenv_str("LOCAL_RANK");

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

        return {};
    }

    const std::optional<std::string> PSM2_CUDA = getenv_str("PSM2_CUDA");

} // namespace shamsys::env
