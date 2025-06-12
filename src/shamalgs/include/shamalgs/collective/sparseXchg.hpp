// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparseXchg.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/mpiErrorCheck.hpp"

namespace shamalgs::collective {

    struct SendPayload {
        i32 receiver_rank;
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct RecvPayload {
        i32 sender_ranks; // should not be plural
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct SparseCommTable {
        std::vector<u64> local_send_vec_comm_ranks;
        std::vector<u64> global_comm_ranks;

        void build(const std::vector<SendPayload> &message_send) {
            StackEntry stack_loc{};

            local_send_vec_comm_ranks.resize(message_send.size());

            i32 iterator = 0;
            for (u64 i = 0; i < message_send.size(); i++) {
                local_send_vec_comm_ranks[i]
                    = sham::pack32(shamcomm::world_rank(), message_send[i].receiver_rank);
            }

            vector_allgatherv(local_send_vec_comm_ranks, global_comm_ranks, MPI_COMM_WORLD);
        }
    };

    inline void sparse_comm_c(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        // send step
        u32 send_idx = 0;
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.x() == shamcomm::world_rank()) {

                auto &payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                MPICHECK(MPI_Isend(
                    payload->get_ptr(),
                    payload->get_size(),
                    MPI_BYTE,
                    comm_ranks.y(),
                    i,
                    MPI_COMM_WORLD,
                    &rq));

                send_idx++;
            }
        }

        // recv step
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.y() == shamcomm::world_rank()) {

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                MPI_Status st;
                i32 cnt;
                MPICHECK(MPI_Probe(comm_ranks.x(), i, MPI_COMM_WORLD, &st));
                MPICHECK(MPI_Get_count(&st, MPI_BYTE, &cnt));

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, dev_sched);

                MPICHECK(MPI_Irecv(
                    payload.payload->get_ptr(),
                    cnt,
                    MPI_BYTE,
                    comm_ranks.x(),
                    i,
                    MPI_COMM_WORLD,
                    &rq));

                message_recv.push_back(std::move(payload));
            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        MPICHECK(MPI_Waitall(rqs.size(), rqs.data(), st_lst.data()));
    }

    inline void base_sparse_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv) {
        StackEntry stack_loc{};

        SparseCommTable comm_table;

        comm_table.build(message_send);

        sparse_comm_c(dev_sched, message_send, message_recv, comm_table);
    }

} // namespace shamalgs::collective
