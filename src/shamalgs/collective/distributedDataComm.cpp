// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file distributedDataComm.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "distributedDataComm.hpp"

#include "shamalgs/serialize.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include <vector>

namespace shamalgs::collective {

    namespace details {
        struct DataTmp {
            u64 sender;
            u64 receiver;
            u64 length;
            std::unique_ptr<sycl::buffer<u8>> &data;

            u64 get_ser_sz() {
                return SerializeHelper::serialize_byte_size<u64>() * 3 +
                       SerializeHelper::serialize_byte_size<u8>(length);
            }
        };

        auto serialize_group_data(std::map<std::pair<i32, i32>, std::vector<DataTmp>> &send_data)
            -> std::map<std::pair<i32, i32>, SerializeHelper> {

            StackEntry stack_loc{};

            std::map<std::pair<i32, i32>, SerializeHelper> serializers;

            for (auto &[key, vect] : send_data) {
                u64 byte_sz = SerializeHelper::serialize_byte_size<u64>(); // vec length
                for (DataTmp &d : vect) {
                    byte_sz += d.get_ser_sz();
                }
                serializers[key].allocate(byte_sz);
            }

            for (auto &[key, vect] : send_data) {
                SerializeHelper &ser = serializers[key];
                ser.write<u64>(vect.size());
                for (DataTmp &d : vect) {
                    ser.write(d.sender);
                    ser.write(d.receiver);
                    ser.write(d.length);
                    ser.write_buf(shambase::get_check_ref(d.data), d.length);
                }
            }

            return serializers;
        }

    } // namespace details

    using SerializedDDataComm = shambase::DistributedDataShared<std::unique_ptr<sycl::buffer<u8>>>;

    void distributed_data_sparse_comm(SerializedDDataComm &send_distrib_data,
                                      SerializedDDataComm &recv_distrib_data,
                                      shamcomm::CommunicationProtocol prot,
                                      std::function<i32(u64)> rank_getter,
                                      std::optional<SparseCommTable> comm_table) {

        StackEntry stack_loc{};

        using namespace shambase;
        using namespace shamsys;
        using DataTmp = details::DataTmp;

        // prepare map
        std::map<std::pair<i32, i32>, std::vector<DataTmp>> send_data;
        send_distrib_data.for_each(
            [&](u64 sender, u64 receiver, std::unique_ptr<sycl::buffer<u8>> &buf) {
                std::pair<i32, i32> key = {rank_getter(sender), rank_getter(receiver)};

                send_data[key].push_back(DataTmp{sender, receiver, get_check_ref(buf).size(), buf});
            });

        // serialize together similar communications
        std::map<std::pair<i32, i32>, SerializeHelper> serializers =
            details::serialize_group_data(send_data);

        // recover bufs from serializers
        std::map<std::pair<i32, i32>, std::unique_ptr<sycl::buffer<u8>>> send_bufs;
        {NamedStackEntry stack_loc2{"recover bufs"};
        for (auto &[key, ser] : serializers) {
            send_bufs[key] = ser.finalize();
        }
        }

        // prepare payload
        std::vector<SendPayload> send_payoad;
        {NamedStackEntry stack_loc2{"prepare payload"};
        for (auto &[key, buf] : send_bufs) {
            send_payoad.push_back(
                {key.second, std::make_unique<shamcomm::CommunicationBuffer>(get_check_ref(buf), prot)});
        }    
        }
        

        // sparse comm
        std::vector<RecvPayload> recv_payload;

        if (comm_table) {
            sparse_comm_c(send_payoad, recv_payload, prot, *comm_table);
        } else {
            base_sparse_comm(send_payoad, recv_payload, prot);
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

                sycl::buffer<u8> buf = shamcomm::CommunicationBuffer::convert(std::move(comm_buf));

                recv_payload_bufs.push_back(RecvPayloadSer{
                    payload.sender_ranks,
                    SerializeHelper(std::make_unique<sycl::buffer<u8>>(std::move(buf)))});
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
                            throw make_except_with_loc<std::runtime_error>("the rank do not matches");
                        }
                    }

                    auto it = recv_distrib_data.add_obj(
                        sender, receiver, std::make_unique<sycl::buffer<u8>>(length));

                    recv.ser.load_buf(*it->second, length);
                }
            }
        }
    }

} // namespace shamalgs::collective