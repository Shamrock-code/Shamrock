// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file worldInfo.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shamcomm/worldInfo.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/mpi.hpp"

namespace shamcomm {

    i32 _world_rank;

    i32 _world_size;

    const i32 world_size(){
        return _world_size;
    }

    const i32 world_rank(){
        return _world_rank;
    }

    void fetch_world_info(){

        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &_world_size));
        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank));

    }

} // namespace shamcomm