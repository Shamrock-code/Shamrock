// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
#include "shambase/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"

namespace shamcomm {

    StateMPI_Aware mpi_cuda_aware;
    StateMPI_Aware mpi_rocm_aware;

    bool forced_on;

    void fetch_mpi_capabilities(bool force_aware) {
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

        if(force_aware){
            forced_on = true;
            mpi_cuda_aware = Yes;
            mpi_rocm_aware = Yes;
        }
    }

    void print_mpi_capabilities() {
        using namespace shambase::term_colors;
        if(!forced_on){
            if (mpi_cuda_aware == Yes) {
                logs::print_ln(" - MPI CUDA-AWARE :", col8b_green() + "Yes" + reset());
            } else if (mpi_cuda_aware == No) {
                logs::print_ln(" - MPI CUDA-AWARE :", col8b_red() + "No" + reset());
            } else if (mpi_cuda_aware == Unknown) {
                logs::print_ln(" - MPI CUDA-AWARE :", col8b_yellow() + "Unknown" + reset());
            }

            if (mpi_rocm_aware == Yes) {
                logs::print_ln(" - MPI ROCM-AWARE :", col8b_green() + "Yes" + reset());
            } else if (mpi_rocm_aware == No) {
                logs::print_ln(" - MPI ROCM-AWARE :", col8b_red() + "No" + reset());
            } else if (mpi_rocm_aware == Unknown) {
                logs::print_ln(" - MPI ROCM-AWARE :", col8b_yellow() + "Unknown" + reset());
            }
        }else {
            logs::print_ln(" - MPI CUDA-AWARE :", col8b_yellow() + "Forced Yes" + reset());
            logs::print_ln(" - MPI ROCM-AWARE :", col8b_yellow() + "Forced Yes" + reset());
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