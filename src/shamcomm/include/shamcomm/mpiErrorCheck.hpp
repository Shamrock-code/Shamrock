// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file mpiErrorCheck.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Use this header to include MPI properly
 *
 */

namespace shamcomm {    
    
    /**
    * @brief check a mpi return code
    */
    void check_mpi_return(int ret, const char* log);

} // namespace shamcomm

#define MPICHECK(mpicall) shamcomm::check_mpi_return(mpicall, #mpicall)