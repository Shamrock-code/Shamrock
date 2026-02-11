// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparse_exchange.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include <optional>
#include <vector>

namespace shamalgs::collective {

    struct CommMessageBufOffset {
        size_t buf_id;
        size_t data_offset;

        friend bool operator==(const CommMessageBufOffset &a, const CommMessageBufOffset &b) {
            return a.buf_id == b.buf_id && a.data_offset == b.data_offset;
        }
        friend bool operator!=(const CommMessageBufOffset &a, const CommMessageBufOffset &b) {
            return !(a == b);
        }
    };

    struct CommMessageInfo {
        size_t message_size;            ///< Size of the MPI message
        i32 rank_sender;                ///< Rank of the sender
        i32 rank_receiver;              ///< Rank of the receiver
        std::optional<i32> message_tag; ///< Tag of the MPI message

        std::optional<CommMessageBufOffset>
            message_bytebuf_offset_send; ///< Offset of the MPI message in the send buffer
        std::optional<CommMessageBufOffset>
            message_bytebuf_offset_recv; ///< Offset of the MPI message in the recv buffer
    };

    struct CommTable {
        std::vector<CommMessageInfo> messages_send; ///< Messages to send
        std::vector<CommMessageInfo> message_all; ///< All messages = (allgatherv of messages_send)
        std::vector<CommMessageInfo> messages_recv;  ///< Messages to recv
        std::vector<size_t> send_message_global_ids; ///< ids of messages_send in message_all
        std::vector<size_t> recv_message_global_ids; ///< ids of messages_recv in message_all

        std::vector<size_t> send_total_sizes; ///< Total size of the send buffer
        std::vector<size_t> recv_total_sizes; ///< Total size of the recv buffer
    };

    CommTable build_sparse_exchange_table(
        const std::vector<CommMessageInfo> &messages_send, size_t max_alloc_size);

    template<sham::USMKindTarget target>
    void sparse_exchange(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_recv,
        const CommTable &comm_table);

} // namespace shamalgs::collective
