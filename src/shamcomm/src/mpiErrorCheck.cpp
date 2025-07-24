// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file mpiErrorCheck.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Use this header to include MPI properly
 *
 */

#include "shamcomm/mpiErrorCheck.hpp"
#include "mpi.h"
#include <cstdio>

void shamcomm::check_mpi_return(int ret, const char *log) {

    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "error in MPI call : %s\n", log);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }
}
