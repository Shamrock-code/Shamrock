// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file patchdata_field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>



#include "aliases.hpp"

#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"

#include "shamrock/legacy/utils/sycl_vector_utils.hpp"

#include "shamrock/patch/PatchDataField.hpp"

namespace patchdata_field {

    enum comm_type {
        CopyToHost, DirectGPU
    };
    enum op_type{
        Send,Recv_Probe
    };

    extern comm_type current_mode; // point this one to the same one in sycl_mpi_interop


    template<class T>
    struct PatchDataFieldMpiRequest{
        MPI_Request mpi_rq;
        comm_type comm_mode;
        op_type comm_op;
        T* comm_ptr;
        u32 comm_val_cnt;
        PatchDataField<T> &pdat_field;


        PatchDataFieldMpiRequest<T> (
            PatchDataField<T> &pdat_field,
            comm_type comm_mode,
            op_type comm_op,
            u32 comm_val_cnt
            );

        inline T* get_mpi_ptr(){
            return comm_ptr;
        }

        void finalize();
    };


    template<class T>
    inline u64 isend( PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){

        rq_lst.emplace_back(p,current_mode,Send,p.size());
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::isend(rq.get_mpi_ptr(), p.size(), get_mpi_type<T>(), rank_dest, tag, comm, &(rq_lst[rq_index].mpi_rq));
        
        return sizeof(T)*p.size();
    }


    

    template<class T>
    inline u64 irecv_probe(PatchDataField<T> &p, std::vector<PatchDataFieldMpiRequest<T>> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
        MPI_Status st;
        i32 cnt;
        mpi::probe(rank_source, tag,comm, & st);
        mpi::get_count(&st, get_mpi_type<T>(), &cnt);

        u32 val_cnt = cnt;



        rq_lst.emplace_back(p,current_mode,Recv_Probe,val_cnt);
            
        u32 rq_index = rq_lst.size() - 1;

        auto & rq = rq_lst[rq_index];   

        mpi::irecv(rq.get_mpi_ptr(), val_cnt, get_mpi_type<T>(), rank_source, tag, comm, &(rq_lst[rq_index].mpi_rq));

        return sizeof(T)*cnt;
    }

    template<class T> 
    inline std::vector<MPI_Request> get_rqs(std::vector<PatchDataFieldMpiRequest<T>> &rq_lst){
        std::vector<MPI_Request> addrs;

        for(auto a : rq_lst){
            addrs.push_back(a.mpi_rq);
        }

        return addrs;
    }

    template<class T>
    inline void waitall(std::vector<PatchDataFieldMpiRequest<T>> &rq_lst){
        std::vector<MPI_Request> addrs;

        for(auto a : rq_lst){
            addrs.push_back(a.mpi_rq);
        }

        std::vector<MPI_Status> st_lst(addrs.size());
        mpi::waitall(addrs.size(), addrs.data(), st_lst.data());

        for(auto a : rq_lst){
            a.finalize();
        }
    }



    template<class T>
    inline void file_write(MPI_File fh, PatchDataField<T> &p){
        MPI_Status st;

        PatchDataFieldMpiRequest<T> rq (p, current_mode, Send,p.size());

        mpi::file_write(fh, rq.get_mpi_ptr(),  p.size(), get_mpi_type<T>(), &st);

        rq.finalize();
    }
}




