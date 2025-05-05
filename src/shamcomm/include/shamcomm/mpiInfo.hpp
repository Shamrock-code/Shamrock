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
 * @file mpiInfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Provide information about MPI capabilities
 *
 * This header provides information about the MPI capabilities of the
 * current process.
 */

#include <optional>
#include <string>
namespace shamcomm {

    /**
     * @brief Enum to describe the MPI capabilities
     */
    enum StateMPI_Aware {
        /**
         * @brief The MPI implementation does not if the feature is supported
         */
        Unknown,
        /**
         * @brief The MPI implementation supports the feature
         */
        Yes,
        /**
         * @brief The MPI implementation does not support the feature
         */
        No,
        /**
         * @brief Feature forced on by the user
         */
        ForcedYes,
        /**
         * @brief Feature forced off by the user
         */
        ForcedNo
    };

    inline bool is_direct_comm_aware(const StateMPI_Aware &v) {
        return (v == Yes || v == ForcedYes);
    }

    /**
     * @brief Get the MPI CUDA aware capability
     *
     * This function returns the MPI CUDA aware capability of the current
     * process. If the capability has not been fetched yet, it raises a
     * std::runtime_error
     *
     * @return The MPI CUDA aware capability
     */
    StateMPI_Aware get_mpi_cuda_aware_status();

    /**
     * @brief Get the MPI ROCM aware capability
     *
     * This function returns the MPI ROCM aware capability of the current
     * process. If the capability has not been fetched yet, it raises a
     * std::runtime_error
     *
     * @return The MPI ROCM aware capability
     */
    StateMPI_Aware get_mpi_rocm_aware_status();

    /// Should DGPU should be forced
    std::optional<StateMPI_Aware> should_force_dgpu_state();

    /**
     * @brief Fetch the MPI capabilities
     *
     * This function fetches the MPI capabilities of the current process.
     *
     * @param forced_state Force the MPI CUDA & ROCM aware capability to be reported as @c ForcedOn
     */
    void fetch_mpi_capabilities(std::optional<StateMPI_Aware> forced_state);

    /**
     * @brief Print the MPI capabilities
     */
    void print_mpi_capabilities();

    /**
     * @brief Get the process name
     *
     * @return std::string the process name
     */
    std::string get_process_name();

} // namespace shamcomm
