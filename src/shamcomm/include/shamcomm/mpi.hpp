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
 * @file mpi.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Use this header to include MPI properly
 *
 */

#if __has_include(<mpi.h>)
    #include <mpi.h>
#elif __has_include(<mpi/mpi.h>) // on the github CI pipeline
    #include <mpi/mpi.h>
#else
    #error "mpi headers cannot be found check the output of "
#endif

#if __has_include(<mpi-ext.h>)
    #include <mpi-ext.h>
    #define FOUND_MPI_EXT
#endif
