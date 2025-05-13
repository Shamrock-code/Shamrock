// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file mpiInfo.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Use this header to include MPI properly
 *
 */

#include "shamcomm/mpiInfo.hpp"
#include "fmt/core.h"
#include "shamcmdopt/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"

namespace shamcomm {

    /**
     * @brief MPI CUDA aware capability
     *
     * This variable contains the MPI CUDA aware capability of the current
     * process. It is set to `Unknown` at the beginning and is updated by the
     * `fetch_mpi_capabilities` function.
     */
    StateMPI_Aware mpi_cuda_aware = StateMPI_Aware::Unknown;

    /**
     * @brief MPI ROCm aware capability
     *
     * This variable contains the MPI ROCm aware capability of the current
     * process. It is set to `Unknown` at the beginning and is updated by the
     * `fetch_mpi_capabilities` function.
     */
    StateMPI_Aware mpi_rocm_aware = StateMPI_Aware::Unknown;

    /**
     * @brief Has the MPI capabilities been fetched?
     *
     * This variable is set to `true` once the `fetch_mpi_capabilities` function
     * has been called.
     */
    bool fetched = false;

    StateMPI_Aware get_mpi_cuda_aware_status() {
        if (!fetched) {
            shambase::throw_with_loc<std::runtime_error>(
                "MPI capabilities have not been fetched yet");
        }
        return mpi_cuda_aware;
    }

    StateMPI_Aware get_mpi_rocm_aware_status() {
        if (!fetched) {
            shambase::throw_with_loc<std::runtime_error>(
                "MPI capabilities have not been fetched yet");
        }
        return mpi_rocm_aware;
    }

    std::optional<StateMPI_Aware> _forced_state;

    void fetch_mpi_capabilities(std::optional<StateMPI_Aware> forced_state) {

        _forced_state = forced_state;

        logs::debug_ln("Comm", "fetching mpi capabilities...");
#ifdef FOUND_MPI_EXT
        logs::debug_mpi_ln("Comm", "FOUND_MPI_EXT is defined");
    // detect MPI cuda aware
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
        logs::debug_mpi_ln("Comm", "MPIX_CUDA_AWARE_SUPPORT is defined");
        if (1 == MPIX_Query_cuda_support()) {
            mpi_cuda_aware = Yes;
        } else {
            mpi_cuda_aware = No;
        }
    #else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        mpi_cuda_aware = Unknown;
    #endif /* MPIX_CUDA_AWARE_SUPPORT */

    // detect MPI rocm aware
    #if defined(MPIX_ROCM_AWARE_SUPPORT)
        logs::debug_mpi_ln("Comm", "MPIX_ROCM_AWARE_SUPPORT is defined");
        if (1 == MPIX_Query_rocm_support()) {
            mpi_rocm_aware = Yes;
        } else {
            mpi_rocm_aware = No;
        }
    #else  /* !defined(MPIX_ROCM_AWARE_SUPPORT) */
        mpi_rocm_aware = Unknown;
    #endif /* MPIX_ROCM_AWARE_SUPPORT */
#else
        mpi_cuda_aware = Unknown;
        mpi_rocm_aware = Unknown;
#endif

        if (forced_state) {
            mpi_cuda_aware = *forced_state;
            mpi_rocm_aware = *forced_state;
        }

        fetched = true;
    }

    std::optional<StateMPI_Aware> should_force_dgpu_state() { return _forced_state; }

    void print_mpi_capabilities() {
        using namespace shambase::term_colors;

        auto print_state = [](std::string log, StateMPI_Aware state) {
            switch (mpi_cuda_aware) {
            case Yes: logs::print_ln(" - " + log + " :", col8b_green() + "Yes" + reset()); break;
            case No: logs::print_ln(" - " + log + " :", col8b_red() + "No" + reset()); break;
            case Unknown:
                logs::print_ln(" - " + log + " :", col8b_yellow() + "Unknown" + reset());
                break;
            case ForcedYes:
                logs::print_ln(" - " + log + " :", col8b_yellow() + "Forced Yes" + reset());
                break;
            case ForcedNo:
                logs::print_ln(" - " + log + " :", col8b_yellow() + "Forced No" + reset());
                break;
            }
        };

        print_state("MPI CUDA-AWARE", mpi_cuda_aware);
        print_state("MPI ROCM-AWARE", mpi_rocm_aware);

        if (_forced_state) {
            print_state("MPI Forced DGPU", *_forced_state);
        }
    }

    std::string get_process_name() {

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;

        int err_code = MPI_Get_processor_name(processor_name, &name_len);

        if (err_code != MPI_SUCCESS) {
            shambase::throw_with_loc<std::runtime_error>("failed getting the process name");
        }

        return {processor_name};
    }
} // namespace shamcomm
