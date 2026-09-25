// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MPIInitGuard.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RAII wrapper around MPI_Init and MPI_Finalize
 */

#include "shamcomm/MPIInitGuard.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include <exception>

namespace shamcomm {

    namespace {

        void log_mpi_init_call(int *argc, char ***argv) {
            logs::debug_mpi_ln("MPIInitGuard", "MPI_Init(", argc, ",", argv, ")");

            if (argc == nullptr) {
                return;
            }

            logs::debug_mpi_ln("MPIInitGuard", "MPI_Init *argc =", *argc);

            if (argv == nullptr || *argv == nullptr) {
                return;
            }

            for (int i = 0; i < *argc; i++) {
                const char *arg = (*argv)[i];
                logs::debug_mpi_ln(
                    "MPIInitGuard", "MPI_Init argv[", i, "] =", (arg != nullptr) ? arg : "nullptr");
            }
        }

    } // namespace

    MPIInitGuard::MPIInitGuard(int *argc, char ***argv) {
        int initialized = 0;
        MPICHECK(MPI_Initialized(&initialized));
        if (initialized) {
            owns_mpi = false;
            return;
        }

        log_mpi_init_call(argc, argv);
        MPICHECK(MPI_Init(argc, argv));
        owns_mpi = true;
    }

    MPIInitGuard::~MPIInitGuard() {
        if (!owns_mpi) {
            return;
        }
        if (std::uncaught_exceptions() > 0) {
            logs::debug_mpi_ln("MPIInitGuard", "MPI_Abort(MPI_COMM_WORLD, 1)");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        close();
    }

    void MPIInitGuard::close() {
        if (!owns_mpi) {
            return;
        }

        int finalized = 0;
        MPICHECK(MPI_Finalized(&finalized));
        if (!finalized) {
            logs::debug_mpi_ln("MPIInitGuard", "MPI_Finalize()");
            MPICHECK(MPI_Finalize());
        }

        owns_mpi = false;
    }

    bool MPIInitGuard::is_active() const { return owns_mpi; }

} // namespace shamcomm
