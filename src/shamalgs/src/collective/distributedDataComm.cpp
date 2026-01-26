// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file distributedDataComm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/distributedDataComm.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/sparse_exchange.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shamcmdopt/env.hpp"
#include <memory>
#include <vector>

auto SPARSE_COMM_MODE = shamcmdopt::getenv_str_default_register(
    "SPARSE_COMM_MODE", "new", "Sparse communication mode (new=with cache, old=without cache)");

namespace {
    struct SparseCommMode {
        enum Mode { NEW, OLD };
    };

    constexpr auto parse_sparse_comm_mode = []() {
        if (SPARSE_COMM_MODE == "new") {
            return SparseCommMode::NEW;
        } else if (SPARSE_COMM_MODE == "old") {
            return SparseCommMode::OLD;
        } else {
            throw std::invalid_argument(
                "Invalid sparse communication mode, valid modes are: new, old");
        }
    };

    bool use_old_sparse_comm_mode = parse_sparse_comm_mode() == SparseCommMode::OLD;

    bool warning_printed = false;
} // namespace

namespace shamalgs::collective {

    namespace details {
        struct DataTmp {
            u64 sender;
            u64 receiver;
            u64 length;
            sham::DeviceBuffer<u8> &data;

            SerializeSize get_ser_sz() {
                return SerializeHelper::serialize_byte_size<u64>() * 3
                       + SerializeHelper::serialize_byte_size<u8>(length);
            }
        };

        auto serialize_group_data(
            std::shared_ptr<sham::DeviceScheduler> dev_sched,
            std::map<std::pair<i32, i32>, std::vector<DataTmp>> &send_data)
            -> std::map<std::pair<i32, i32>, SerializeHelper> {

            StackEntry stack_loc{};

            std::map<std::pair<i32, i32>, SerializeHelper> serializers;

            for (auto &[key, vect] : send_data) {
                SerializeSize byte_sz = SerializeHelper::serialize_byte_size<u64>(); // vec length
                for (DataTmp &d : vect) {
                    byte_sz += d.get_ser_sz();
                }
                serializers.emplace(key, dev_sched);
                serializers.at(key).allocate(byte_sz);
            }

            for (auto &[key, vect] : send_data) {
                SerializeHelper &ser = serializers.at(key);
                ser.write<u64>(vect.size());
                for (DataTmp &d : vect) {
                    ser.write(d.sender);
                    ser.write(d.receiver);
                    ser.write(d.length);
                    ser.write_buf(d.data, d.length);
                }
            }

            return serializers;
        }

    } // namespace details

    void distributed_data_sparse_comm_old(
        sham::DeviceScheduler_ptr dev_sched,
        SerializedDDataComm &send_distrib_data,
        SerializedDDataComm &recv_distrib_data,
        std::function<i32(u64)> rank_getter,
        std::optional<SparseCommTable> comm_table) {

        StackEntry stack_loc{};

        using namespace shambase;
        using DataTmp = details::DataTmp;

        // prepare map
        std::map<std::pair<i32, i32>, std::vector<DataTmp>> send_data;
        send_distrib_data.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u8> &buf) {
            std::pair<i32, i32> key = {rank_getter(sender), rank_getter(receiver)};

            send_data[key].push_back(DataTmp{sender, receiver, buf.get_size(), buf});
        });

        // serialize together similar communications
        std::map<std::pair<i32, i32>, SerializeHelper> serializers
            = details::serialize_group_data(dev_sched, send_data);

        // recover bufs from serializers
        std::map<std::pair<i32, i32>, std::unique_ptr<sham::DeviceBuffer<u8>>> send_bufs;
        {
            NamedStackEntry stack_loc2{"recover bufs"};
            for (auto &[key, ser] : serializers) {
                send_bufs[key] = std::make_unique<sham::DeviceBuffer<u8>>(ser.finalize());
            }
        }

        // prepare payload
        std::vector<SendPayload> send_payoad;
        {
            NamedStackEntry stack_loc2{"prepare payload"};
            for (auto &[key, buf] : send_bufs) {
                send_payoad.push_back(
                    {key.second,
                     std::make_unique<shamcomm::CommunicationBuffer>(
                         shambase::extract_pointer(buf), dev_sched)});
            }
        }

        // sparse comm
        std::vector<RecvPayload> recv_payload;

        if (comm_table) {
            sparse_comm_c(dev_sched, send_payoad, recv_payload, *comm_table);
        } else {
            base_sparse_comm(dev_sched, send_payoad, recv_payload);
        }

        // make serializers from recv buffs
        struct RecvPayloadSer {
            i32 sender_ranks;
            SerializeHelper ser;
        };

        std::vector<RecvPayloadSer> recv_payload_bufs;

        {
            NamedStackEntry stack_loc2{"move payloads"};
            for (RecvPayload &payload : recv_payload) {

                shamcomm::CommunicationBuffer comm_buf = extract_pointer(payload.payload);

                sham::DeviceBuffer<u8> buf
                    = shamcomm::CommunicationBuffer::convert_usm(std::move(comm_buf));

                recv_payload_bufs.push_back(
                    RecvPayloadSer{
                        payload.sender_ranks, SerializeHelper(dev_sched, std::move(buf))});
            }
        }

        {
            NamedStackEntry stack_loc2{"split recv comms"};
            // deserialize into the shared distributed data
            for (RecvPayloadSer &recv : recv_payload_bufs) {
                u64 cnt_obj;
                recv.ser.load(cnt_obj);
                for (u32 i = 0; i < cnt_obj; i++) {
                    u64 sender, receiver, length;

                    recv.ser.load(sender);
                    recv.ser.load(receiver);
                    recv.ser.load(length);

                    { // check correctness ranks
                        i32 supposed_sender_rank = rank_getter(sender);
                        i32 real_sender_rank     = recv.sender_ranks;
                        if (supposed_sender_rank != real_sender_rank) {
                            throw make_except_with_loc<std::runtime_error>(
                                "the rank do not matches");
                        }
                    }

                    auto it = recv_distrib_data.add_obj(
                        sender, receiver, sham::DeviceBuffer<u8>(length, dev_sched));

                    recv.ser.load_buf(it->second, length);
                }
            }
        }
    }

    void distributed_data_sparse_comm(
        sham::DeviceScheduler_ptr dev_sched,
        SerializedDDataComm &send_distrib_data,
        SerializedDDataComm &recv_distrib_data,
        std::function<i32(u64)> rank_getter,
        DDSCommCache &cache,
        std::optional<SparseCommTable> comm_table) {

        if (use_old_sparse_comm_mode) {
            if (shamcomm::world_rank() == 0 && !warning_printed) {
                logger::warn_ln("SparseComm", "using old sparse communication mode");
                warning_printed = true;
            }
            return distributed_data_sparse_comm_old(
                dev_sched, send_distrib_data, recv_distrib_data, rank_getter, comm_table);
        }

        __shamrock_stack_entry();

        using namespace shambase;
        using DataTmp = details::DataTmp;

        // prepare map
        std::map<std::pair<i32, i32>, std::vector<DataTmp>> send_data;
        send_distrib_data.for_each([&](u64 sender, u64 receiver, sham::DeviceBuffer<u8> &buf) {
            std::pair<i32, i32> key = {rank_getter(sender), rank_getter(receiver)};

            send_data[key].push_back(DataTmp{sender, receiver, buf.get_size(), buf});
        });

        // serialize together similar communications
        std::map<std::pair<i32, i32>, SerializeHelper> serializers
            = details::serialize_group_data(dev_sched, send_data);

        // recover bufs from serializers
        std::map<std::pair<i32, i32>, std::unique_ptr<sham::DeviceBuffer<u8>>> send_bufs;
        {
            NamedStackEntry stack_loc2{"recover bufs"};
            for (auto &[key, ser] : serializers) {
                send_bufs[key] = std::make_unique<sham::DeviceBuffer<u8>>(ser.finalize());
            }
        }

        std::vector<shamalgs::collective::CommMessageInfo> messages_send;
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> data_send;

        for (auto &[key, buf] : send_bufs) {

            auto [sender, receiver] = key;
            u64 size                = buf->get_size();

            messages_send.push_back(
                shamalgs::collective::CommMessageInfo{
                    size,
                    sender,
                    receiver,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                });

            data_send.push_back(std::move(buf));
        }

        size_t max_alloc_size;
        if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
            max_alloc_size = dev_sched->ctx->device->prop.max_mem_alloc_size_dev;
        } else {
            max_alloc_size = dev_sched->ctx->device->prop.max_mem_alloc_size_host;
        }
        max_alloc_size *= 0.95; // keep 5% of the max alloc size for safety

        shamalgs::collective::CommTable comm_table2
            = shamalgs::collective::build_sparse_exchange_table(messages_send, max_alloc_size);

        if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
            cache.set_sizes<sham::device>(
                dev_sched, comm_table2.send_total_sizes, comm_table2.recv_total_sizes);
        } else {
            cache.set_sizes<sham::host>(
                dev_sched, comm_table2.send_total_sizes, comm_table2.recv_total_sizes);
        }

        if (comm_table2.messages_send.size() != data_send.size()) {
            std::vector<size_t> tmp1{};
            for (size_t i = 0; i < data_send.size(); i++) {
                tmp1.push_back(comm_table2.messages_send[i].message_size);
            }

            std::vector<size_t> tmp2{};
            for (size_t i = 0; i < data_send.size(); i++) {
                tmp2.push_back(data_send[i]->get_size());
            }

            throw make_except_with_loc<std::runtime_error>(
                shambase::format("message send mismatch : {} != {}", tmp1, tmp2));
        }

        if (comm_table2.messages_send.size() != messages_send.size()) {
            std::vector<size_t> tmp1{};
            for (size_t i = 0; i < comm_table2.messages_send.size(); i++) {
                tmp1.push_back(comm_table2.messages_send[i].message_size);
            }

            std::vector<size_t> tmp2{};
            for (size_t i = 0; i < messages_send.size(); i++) {
                tmp2.push_back(messages_send[i].message_size);
            }
            throw make_except_with_loc<std::runtime_error>(
                shambase::format("message send mismatch : {} != {}", tmp1, tmp2));
        }

        for (size_t i = 0; i < comm_table2.messages_send.size(); i++) {
            auto &msg_info   = comm_table2.messages_send[i];
            auto offset_info = shambase::get_check_ref(msg_info.message_bytebuf_offset_send);
            auto &buf_src    = shambase::get_check_ref(data_send.at(i));

            SHAM_ASSERT(buf_src.get_size() == msg_info.message_size);

            cache.send_cache_write_buf_at(offset_info.buf_id, offset_info.data_offset, buf_src);
        }

        if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
            shamalgs::collective::sparse_exchange<sham::device>(
                dev_sched,
                cache.get_cache1<sham::device>(),
                cache.get_cache2<sham::device>(),
                comm_table2);
        } else {
            shamalgs::collective::sparse_exchange<sham::host>(
                dev_sched,
                cache.get_cache1<sham::host>(),
                cache.get_cache2<sham::host>(),
                comm_table2);
        }

        // make serializers from recv buffs
        struct RecvPayloadSer {
            i32 sender_ranks;
            SerializeHelper ser;
        };

        std::vector<RecvPayloadSer> recv_payload_bufs;

        for (auto &msg : comm_table2.messages_recv) {

            u64 size     = msg.message_size;
            i32 sender   = msg.rank_sender;
            i32 receiver = msg.rank_receiver;

            auto offset_info = shambase::get_check_ref(msg.message_bytebuf_offset_recv);

            sham::DeviceBuffer<u8> recov(size, dev_sched);
            cache.recv_cache_read_buf_at(offset_info.buf_id, offset_info.data_offset, size, recov);

            recv_payload_bufs.push_back(
                RecvPayloadSer{sender, SerializeHelper(dev_sched, std::move(recov))});
        }

        {
            NamedStackEntry stack_loc2{"split recv comms"};
            // deserialize into the shared distributed data
            for (RecvPayloadSer &recv : recv_payload_bufs) {
                u64 cnt_obj;
                recv.ser.load(cnt_obj);
                for (u32 i = 0; i < cnt_obj; i++) {
                    u64 sender, receiver, length;

                    recv.ser.load(sender);
                    recv.ser.load(receiver);
                    recv.ser.load(length);

                    { // check correctness ranks
                        i32 supposed_sender_rank = rank_getter(sender);
                        i32 real_sender_rank     = recv.sender_ranks;
                        if (supposed_sender_rank != real_sender_rank) {
                            throw make_except_with_loc<std::runtime_error>(shambase::format(
                                "the rank do not matches {} != {}",
                                supposed_sender_rank,
                                real_sender_rank));
                        }
                    }

                    auto it = recv_distrib_data.add_obj(
                        sender, receiver, sham::DeviceBuffer<u8>(length, dev_sched));

                    recv.ser.load_buf(it->second, length);
                }
            }
        }
    }

} // namespace shamalgs::collective
