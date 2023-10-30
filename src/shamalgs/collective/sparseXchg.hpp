// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparseXchg.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/math.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamcomm/CommunicationBuffer.hpp"
#include "shambase/integer.hpp"
#include "shamsys/legacy/log.hpp"

namespace shamalgs::collective {

    struct SendPayload{
        i32 receiver_rank;
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct RecvPayload{
        i32 sender_ranks; //should not be plural
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct SparseCommTable{
        std::vector<u64> local_send_vec_comm_ranks;
        std::vector<u64> global_comm_ranks;

        void build(const std::vector<SendPayload> & message_send){
            StackEntry stack_loc{};
            using namespace shamsys::instance;

            local_send_vec_comm_ranks.resize(message_send.size());

            i32 iterator = 0;
            for (u64 i = 0; i < message_send.size(); i++) {
                local_send_vec_comm_ranks[i] = sham::pack32(shamcomm::world_rank(), message_send[i].receiver_rank);
            }

            vector_allgatherv(local_send_vec_comm_ranks, global_comm_ranks, MPI_COMM_WORLD);
        }
    };

    inline void sparse_comm_c(const std::vector<SendPayload> & message_send,
        std::vector<RecvPayload> & message_recv,
        shamcomm::CommunicationProtocol protocol,
        const SparseCommTable & comm_table){
        StackEntry stack_loc{};

        using namespace shamsys::instance;

        //share comm list accros nodes
        const std::vector<u64> & send_vec_comm_ranks= comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> & global_comm_ranks = comm_table.global_comm_ranks;


        //note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        //send step
        u32 send_idx = 0;
        for(u32 i = 0; i < global_comm_ranks.size(); i++){
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if(comm_ranks.x() == shamcomm::world_rank()){

                auto & payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto & rq = rqs[rq_index]; 

                mpi::isend(
                    payload->get_ptr(), 
                    payload->get_bytesize(), 
                    MPI_BYTE, 
                    comm_ranks.y(), 
                    i, 
                    MPI_COMM_WORLD, 
                    &rq);

                send_idx++;
            }
        }


        //recv step
        for(u32 i = 0; i < global_comm_ranks.size(); i++){
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if(comm_ranks.y() == shamcomm::world_rank()){

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto & rq = rqs[rq_index];  

                MPI_Status st;
                i32 cnt;
                mpi::probe(comm_ranks.x(), i,MPI_COMM_WORLD, & st);
                mpi::get_count(&st, MPI_BYTE, &cnt);

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, protocol);

                mpi::irecv(
                    payload.payload->get_ptr(), 
                    cnt, 
                    MPI_BYTE, 
                    comm_ranks.x(), 
                    i, 
                    MPI_COMM_WORLD, 
                    &rq);

                message_recv.push_back(std::move(payload));

            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        mpi::waitall(rqs.size(), rqs.data(), st_lst.data());
    }


    inline void base_sparse_comm(
        const std::vector<SendPayload> & message_send,
        std::vector<RecvPayload> & message_recv,
        shamcomm::CommunicationProtocol protocol
        ) 
    {
        StackEntry stack_loc{};

        using namespace shamsys::instance;

        SparseCommTable comm_table;

        comm_table.build(message_send);
        
        sparse_comm_c(message_send, message_recv, protocol, comm_table);
    }

} // namespace shamalgs::collective