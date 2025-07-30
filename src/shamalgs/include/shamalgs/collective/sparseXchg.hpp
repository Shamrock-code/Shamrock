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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <string_view>
#include <functional>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

    void sparse_comm_debug_infos(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table);

    void sparse_comm_isend_probe_count_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table);

    void sparse_comm_allgather_isend_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table);

    inline void sparse_comm_c(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        // sparse_comm_debug_infos(dev_sched, message_send, message_recv, comm_table);
        //    sparse_comm_isend_probe_count_irecv(dev_sched, message_send, message_recv,
        //    comm_table);
        sparse_comm_allgather_isend_irecv(dev_sched, message_send, message_recv, comm_table);
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

    inline void base_sparse_comm_max_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        u32 max_simultaneous_send) {

        int send_loc = message_send.size();
        int send_max_count;
        shamcomm::mpi::Allreduce(&send_loc, &send_max_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // logger::raw_ln(send_loc, send_max_count);

        StackEntry stack_loc{};

        int i = 0;
        while (i < send_max_count) {

            if (i > 0) {
                ON_RANK_0(
                    logger::warn_ln("SparseComm", "Splitted sparse comm", i, "/", send_max_count));
            }

            std::vector<SendPayload> message_send_tmp;
            std::vector<RecvPayload> message_recv_tmp;

            for (int j = i; (j < (i + max_simultaneous_send)) && (j < message_send.size()); j++) {
                // logger::raw_ln("emplace message", j);
                message_send_tmp.emplace_back(std::move(message_send[j]));
            }

            base_sparse_comm(dev_sched, message_send_tmp, message_recv_tmp);

            for (int j = 0; j < message_recv_tmp.size(); j++) {
                message_recv.emplace_back(std::move(message_recv_tmp[j]));
            }

            i += max_simultaneous_send;
        }
    }

} // namespace shamalgs::collective
