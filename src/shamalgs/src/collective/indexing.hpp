// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file indexing.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"

namespace shamalgs::collective {

    struct ViewInfo{
        u64 total_byte_count;
        u64 head_offset;
    };

    
    inline ViewInfo fetch_view(u64 byte_count){

        u64 scan_val;
        mpi::exscan(&byte_count, &scan_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        if(shamcomm::world_rank() == 0){
            scan_val = 0;
        }

        u64 sum_val;
        mpi::allreduce(&byte_count, &sum_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        logger::debug_mpi_ln("fetch view",byte_count, "->",scan_val, "sum:",sum_val);

        return {sum_val,scan_val};
    }

    inline ViewInfo fetch_view_known_total(u64 byte_count,u64 total_byte){

        u64 scan_val;
        mpi::exscan(&byte_count, &scan_val, 1, get_mpi_type<u64>(), MPI_SUM, MPI_COMM_WORLD);

        if(shamcomm::world_rank() == 0){
            scan_val = 0;
        }

        logger::debug_mpi_ln("fetch view",byte_count, "->",scan_val, "sum:",total_byte);

        return {total_byte,scan_val};
    }
    

} // namespace shamalgs::collective