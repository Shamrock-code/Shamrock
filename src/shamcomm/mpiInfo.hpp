// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mpiInfo.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Use this header to include MPI properly
 *
 */

namespace shamcomm {

    enum StateMPI_Aware { Unknown, Yes, No };

    extern StateMPI_Aware mpi_cuda_aware;
    extern StateMPI_Aware mpi_rocm_aware;

    void fetch_mpi_capabilities();

    void print_mpi_capabilities();

} // namespace shamcomm