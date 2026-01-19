// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_exchange.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/sparse_exchange.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/narrowing.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/RequestList.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include <stdexcept>
namespace shamalgs::collective {

    CommMessageInfo unpack(u64_2 comm_info) {
        u64 comm_vec        = comm_info.x();
        size_t message_size = comm_info.y();
        u32_2 comm_ranks    = sham::unpack32(comm_vec);
        u32 sender          = comm_ranks.x();
        u32 receiver        = comm_ranks.y();

        if (message_size == 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "Message size is 0 for rank {}, sender = {}, receiver = {}",
                shamcomm::world_rank(),
                sender,
                receiver));
        }

        return CommMessageInfo{
            message_size,
            static_cast<i32>(sender),
            static_cast<i32>(receiver),
            std::nullopt,
            std::nullopt,
            std::nullopt};
    };

    /// fetch u64_2 from global message data
    std::vector<u64_2> fetch_global_message_data(
        const std::vector<CommMessageInfo> &messages_send) {

        std::vector<u64_2> local_data = std::vector<u64_2>(messages_send.size());

        for (size_t i = 0; i < messages_send.size(); i++) {
            u32 sender          = static_cast<u32>(messages_send[i].rank_sender);
            u32 receiver        = static_cast<u32>(messages_send[i].rank_receiver);
            size_t message_size = messages_send[i].message_size;

            if (sender != shamcomm::world_rank()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "You are trying to send a message from a rank that does not posses it\n"
                    "    sender = {}, receiver = {}, world_rank = {}",
                    sender,
                    receiver,
                    shamcomm::world_rank()));
            }

            local_data[i] = u64_2{sham::pack32(sender, receiver), message_size};
        }

        std::vector<u64_2> global_data;
        vector_allgatherv(local_data, global_data, MPI_COMM_WORLD);

        return global_data; // there should be return value optimisation here
    }

    /// decode message to get message
    std::vector<CommMessageInfo> decode_all_message(const std::vector<u64_2> &global_data) {
        std::vector<CommMessageInfo> message_all(global_data.size());
        for (u64 i = 0; i < global_data.size(); i++) {
            message_all[i] = unpack(global_data[i]);
        }

        return message_all;
    }

    /// compute message tags
    void compute_tags(std::vector<CommMessageInfo> &message_all) {

        std::vector<i32> tag_map(shamcomm::world_size(), 0);

        for (u64 i = 0; i < message_all.size(); i++) {
            auto &message_info = message_all[i];
            auto sender        = message_info.rank_sender;

            // tagging logic
            i32 &tag_map_ref = tag_map[static_cast<size_t>(sender)];
            i32 tag          = tag_map_ref;
            tag_map_ref++;

            message_info.message_tag = tag;
        }
    }

    CommTable build_sparse_exchange_table(
        const std::vector<CommMessageInfo> &messages_send, size_t max_alloc_size) {
        __shamrock_stack_entry();

        std::vector<u64_2> global_data = fetch_global_message_data(messages_send);

        std::vector<CommMessageInfo> message_all = decode_all_message(global_data);

        compute_tags(message_all);

        ////////////////////////////////////////////////////////////
        // Compute offsets
        ////////////////////////////////////////////////////////////

        std::vector<size_t> send_buf_sizes{};
        std::vector<size_t> recv_buf_sizes{};

        u32 send_idx = 0;
        u32 recv_idx = 0;
        {
            size_t tmp_recv_offset = 0;
            size_t tmp_send_offset = 0;
            size_t send_buf_id     = 0;
            size_t recv_buf_id     = 0;
            for (u64 i = 0; i < message_all.size(); i++) {
                auto &message_info = message_all[i];

                auto sender   = message_info.rank_sender;
                auto receiver = message_info.rank_receiver;

                // offset logic (& buffer selection)
                if (sender == shamcomm::world_rank()) {
                    if (message_info.message_size > max_alloc_size) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            shambase::format(
                                "Message size is greater than the max alloc size\n"
                                "    message_size = {}, max_alloc_size = {}",
                                message_info.message_size,
                                max_alloc_size));
                    }

                    if (send_buf_sizes.size() == 0) {
                        send_buf_sizes.push_back(0);
                    }

                    if (tmp_send_offset + message_info.message_size > max_alloc_size) {
                        send_buf_id++;
                        tmp_send_offset = 0;
                        send_buf_sizes.push_back(0);
                    }

                    message_info.message_bytebuf_offset_send = {send_buf_id, tmp_send_offset};
                    tmp_send_offset += message_info.message_size;
                    send_buf_sizes.at(send_buf_id) += message_info.message_size;

                    send_idx++;
                }

                if (receiver == shamcomm::world_rank()) {

                    if (message_info.message_size > max_alloc_size) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            shambase::format(
                                "Message size is greater than the max alloc size\n"
                                "    message_size = {}, max_alloc_size = {}",
                                message_info.message_size,
                                max_alloc_size));
                    }

                    if (recv_buf_sizes.size() == 0) {
                        recv_buf_sizes.push_back(0);
                    }

                    if (tmp_recv_offset + message_info.message_size > max_alloc_size) {
                        recv_buf_id++;
                        tmp_recv_offset = 0;
                        recv_buf_sizes.push_back(0);
                    }

                    message_info.message_bytebuf_offset_recv = {recv_buf_id, tmp_recv_offset};
                    tmp_recv_offset += message_info.message_size;
                    recv_buf_sizes.at(recv_buf_id) += message_info.message_size;

                    recv_idx++;
                }

                message_all[i] = message_info;
            }
        }

        ////////////////////////////////////////////////////////////
        // now that all comm were computed we can build the send and recv message lists
        ////////////////////////////////////////////////////////////

        std::vector<CommMessageInfo> ret_message_send(send_idx);
        std::vector<CommMessageInfo> ret_message_recv(recv_idx);

        std::vector<size_t> send_message_global_ids(send_idx);
        std::vector<size_t> recv_message_global_ids(recv_idx);

        send_idx = 0;
        recv_idx = 0;

        for (size_t i = 0; i < message_all.size(); i++) {
            auto message_info = message_all[i];
            if (message_info.rank_sender == shamcomm::world_rank()) {
                ret_message_send[send_idx]        = message_info;
                send_message_global_ids[send_idx] = i;
                send_idx++;
            }
            if (message_info.rank_receiver == shamcomm::world_rank()) {
                ret_message_recv[recv_idx]        = message_info;
                recv_message_global_ids[recv_idx] = i;
                recv_idx++;
            }
        }

        return CommTable{
            ret_message_send,
            message_all,
            ret_message_recv,
            send_message_global_ids,
            recv_message_global_ids,
            send_buf_sizes,
            recv_buf_sizes};
    }

    void sparse_exchange(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<const u8 *> &bytebuffer_send,
        const std::vector<u8 *> &bytebuffer_recv,
        const CommTable &comm_table) {

        __shamrock_stack_entry();

        u32 SHAM_SPARSE_COMM_INFLIGHT_LIM = 128; // TODO: use the env variable

        RequestList rqs;
        for (size_t i = 0; i < comm_table.message_all.size(); i++) {

            auto message_info = comm_table.message_all[i];

            if (message_info.rank_sender == shamcomm::world_rank()) {
                auto off_info = shambase::get_check_ref(message_info.message_bytebuf_offset_send);
                auto ptr      = bytebuffer_send.at(off_info.buf_id) + off_info.data_offset;
                auto &rq      = rqs.new_request();
                shamcomm::mpi::Isend(
                    ptr,
                    shambase::narrow_or_throw<i32>(message_info.message_size),
                    MPI_BYTE,
                    message_info.rank_receiver,
                    shambase::get_check_ref(message_info.message_tag),
                    MPI_COMM_WORLD,
                    &rq);
            }

            if (message_info.rank_receiver == shamcomm::world_rank()) {
                auto off_info = shambase::get_check_ref(message_info.message_bytebuf_offset_recv);
                auto ptr      = bytebuffer_recv.at(off_info.buf_id) + off_info.data_offset;
                auto &rq      = rqs.new_request();
                shamcomm::mpi::Irecv(
                    ptr,
                    shambase::narrow_or_throw<i32>(message_info.message_size),
                    MPI_BYTE,
                    message_info.rank_sender,
                    shambase::get_check_ref(message_info.message_tag),
                    MPI_COMM_WORLD,
                    &rq);
            }

            rqs.spin_lock_partial_wait(SHAM_SPARSE_COMM_INFLIGHT_LIM, 120, 10);
        }
        rqs.wait_all();
    }

    template<sham::USMKindTarget target>
    void sparse_exchange(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_recv,
        const CommTable &comm_table) {

        __shamrock_stack_entry();

        if (&bytebuffer_send == &bytebuffer_recv) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "In-place sparse_exchange is not supported. Send and receive buffers must be "
                "distinct.");
        }

        if (comm_table.send_total_sizes.size() != bytebuffer_send.size()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "The send total size is greater than the send buffer size\n"
                "    send_total_sizes = {}, send_buffer_size = {}",
                comm_table.send_total_sizes.size(),
                bytebuffer_send.size()));
        }

        if (comm_table.recv_total_sizes.size() != bytebuffer_recv.size()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "The recv total size is greater than the recv buffer size\n"
                "    recv_total_sizes = {}, recv_buffer_size = {}",
                comm_table.recv_total_sizes.size(),
                bytebuffer_recv.size()));
        }

        for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
            if (comm_table.send_total_sizes[i] > bytebuffer_send[i]->get_size()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "The send total size is greater than the send buffer size\n"
                    "    send_total_sizes = {}, send_buffer_size = {}, buf_id = {}",
                    comm_table.send_total_sizes[i],
                    bytebuffer_send[i]->get_size(),
                    i));
            }
        }

        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            if (comm_table.recv_total_sizes[i] > bytebuffer_recv[i]->get_size()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "The recv total size is greater than the recv buffer size\n"
                    "    recv_total_sizes = {}, recv_buffer_size = {}, buf_id = {}",
                    comm_table.recv_total_sizes[i],
                    bytebuffer_recv[i]->get_size(),
                    i));
            }
        }

        bool direct_gpu_capable = dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable;

        if (!direct_gpu_capable && target == sham::device) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "You are trying to use a device buffer on the device but the device is not "
                "direct "
                "GPU capable");
        }

        std::vector<const u8 *> send_ptrs(bytebuffer_send.size());
        std::vector<u8 *> recv_ptrs(bytebuffer_recv.size());

        sham::EventList depends_list;
        for (size_t i = 0; i < bytebuffer_send.size(); i++) {
            send_ptrs[i]
                = shambase::get_check_ref(bytebuffer_send[i]).get_read_access(depends_list);
        }

        for (size_t i = 0; i < bytebuffer_recv.size(); i++) {
            recv_ptrs[i]
                = shambase::get_check_ref(bytebuffer_recv[i]).get_write_access(depends_list);
        }
        depends_list.wait();

        sparse_exchange(dev_sched, send_ptrs, recv_ptrs, comm_table);

        for (size_t i = 0; i < bytebuffer_send.size(); i++) {
            shambase::get_check_ref(bytebuffer_send[i]).complete_event_state(sycl::event{});
        }

        for (size_t i = 0; i < bytebuffer_recv.size(); i++) {
            shambase::get_check_ref(bytebuffer_recv[i]).complete_event_state(sycl::event{});
        }
    }

    // template instantiations
    template void sparse_exchange<sham::device>(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::device>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::device>>> &bytebuffer_recv,
        const CommTable &comm_table);

    template void sparse_exchange<sham::host>(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> &bytebuffer_recv,
        const CommTable &comm_table);

} // namespace shamalgs::collective
