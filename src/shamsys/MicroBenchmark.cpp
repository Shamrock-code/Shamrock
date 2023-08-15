// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsys/MicroBenchmark.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/comm/CommunicationBuffer.hpp"
#include "shamsys/legacy/log.hpp"
#include <mpi.h>
#include <stdexcept>



namespace shamsys::microbench{
    void p2p_bandwith(u32 wr_sender, u32 wr_receiv);
    void p2p_latency(u32 wr1, u32 wr2);
} // namespace shamsys::microbench

void shamsys::run_micro_benchmark(){
    StackEntry stack_loc{};

    if(instance::world_rank == 0){
        logger::raw_ln("Running micro benchamrks :");
    }


    u32 wr1 = 0;
    u32 wr2 = instance::world_size-1;

    microbench::p2p_bandwith(wr1, wr2);
    if(instance::world_size > 1){
        microbench::p2p_latency(wr1, wr2);
    }
}

void shamsys::microbench::p2p_bandwith(u32 wr_sender, u32 wr_receiv){
    StackEntry stack_loc{};

    u32 wr = instance::world_rank;

    u64 lenght = 1024UL*1014UL*8UL; //8MB messages
    CommunicationBuffer buf_recv{lenght, shamsys::get_protocol()};
    CommunicationBuffer buf_send{lenght, shamsys::get_protocol()};

    std::vector<MPI_Request> rqs;

    f64 t = 0;
    u64 loops = 0;
    do{
        loops ++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if(wr == wr_sender){
            rqs.push_back(MPI_Request{});
            u32 rq_index = rqs.size() - 1;
            auto & rq = rqs[rq_index]; 
            mpi::isend(buf_send.get_ptr(), lenght, MPI_BYTE, wr_receiv, 0, MPI_COMM_WORLD, &rq);
        }
        if(wr == wr_receiv){
            MPI_Status s;
            mpi::recv(buf_recv.get_ptr(), lenght, MPI_BYTE, wr_sender, 0, MPI_COMM_WORLD, &s);
        }
        std::vector<MPI_Status> st_lst(rqs.size());
        if(rqs.size() > 0){
            mpi::waitall(rqs.size(), rqs.data(), st_lst.data());
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

    }while(shamalgs::collective::allreduce_min(t) < 1);

    if(instance::world_rank == 0){
        logger::raw_ln(shambase::format(" - p2p bandwith : {:0.2f} GB.s^-1 (ranks : {} -> {}) (loops : {})", 
            (f64(lenght*loops) / t)*1e-9 , 
            wr_sender, 
            wr_receiv,
            loops));
    }
}

void shamsys::microbench::p2p_latency(u32 wr1, u32 wr2){
    StackEntry stack_loc{};

    if(wr1 == wr2){
        throw shambase::throw_with_loc<std::invalid_argument>("can not launch this test with same ranks");
    }

    u32 wr = instance::world_rank;

    u64 lenght = 8ULL; //8B messages
    CommunicationBuffer buf_recv{lenght, shamsys::get_protocol()};
    CommunicationBuffer buf_send{lenght, shamsys::get_protocol()};

    f64 t = 0;
    u64 loops = 0;
    do{
        loops ++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if(wr == wr1){
            MPI_Status s;
            mpi::send(buf_send.get_ptr(), lenght, MPI_BYTE, wr2, 0, MPI_COMM_WORLD); 
            mpi::recv(buf_recv.get_ptr(), lenght, MPI_BYTE, wr2, 1, MPI_COMM_WORLD, &s);
        }
        if(wr == wr2){
            MPI_Status s;
            mpi::recv(buf_recv.get_ptr(), lenght, MPI_BYTE, wr1, 0, MPI_COMM_WORLD, &s);
            mpi::send(buf_send.get_ptr(), lenght, MPI_BYTE, wr1, 1, MPI_COMM_WORLD); 
            
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

    }while(shamalgs::collective::allreduce_min(t) < 1);

    if(instance::world_rank == 0){
        logger::raw_ln(shambase::format(" - p2p latency  : {:.4e} s (ranks : {} <-> {}) (loops : {})", 
            t / f64(loops) , 
            wr1, 
            wr2,
            loops));
    }
}