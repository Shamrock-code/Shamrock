// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @brief Utility header to include MPI properly
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
