// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/sparse_exchange.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

namespace {

    struct TestElement {
        i32 sender, receiver;
        u32 size;
    };

} // namespace

void reorder_msg(std::vector<TestElement> &test_elements) {
    std::sort(test_elements.begin(), test_elements.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.sender
               < rhs.sender; //|| (lhs.sender == rhs.sender && lhs.receiver < rhs.receiver);
    });
}

void test_sparse_exchange(std::vector<TestElement> test_elements) {
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    reorder_msg(test_elements);

    std::vector<sham::DeviceBuffer<u8>> all_bufs;

    std::mt19937 eng(0x123);
    for (const auto &test_element : test_elements) {
        all_bufs.push_back(
            shamalgs::random::mock_buffer_usm<u8>(dev_sched, eng(), test_element.size));
    }

    sham::DeviceBuffer<u8> send_buf(0, dev_sched);
    std::vector<shamalgs::collective::CommMessageInfo> messages_send;

    size_t total_recv_size  = 0;
    size_t total_recv_count = 0;
    size_t sender_offset    = 0;
    size_t sender_count     = 0;
    for (u32 i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].sender == shamcomm::world_rank()) {
            messages_send.push_back(
                shamalgs::collective::CommMessageInfo{
                    test_elements[i].size,
                    test_elements[i].sender,
                    test_elements[i].receiver,
                    std::nullopt,
                    sender_offset,
                    std::nullopt,
                });

            logger::info_ln(
                "sparse exchange test",
                "rank :",
                shamcomm::world_rank(),
                "send message : (",
                test_elements[i].sender,
                "->",
                test_elements[i].receiver,
                ") data :",
                all_bufs[i].copy_to_stdvec());

            send_buf.append(all_bufs[i]);
            sender_offset += test_elements[i].size;
            sender_count++;
        }
        if (test_elements[i].receiver == shamcomm::world_rank()) {
            total_recv_size += test_elements[i].size;
            total_recv_count++;
        }
    }

    shamalgs::collective::CommTable comm_table
        = shamalgs::collective::build_sparse_exchange_table(messages_send);

    REQUIRE_EQUAL(comm_table.send_total_size, sender_offset);
    REQUIRE_EQUAL(comm_table.recv_total_size, total_recv_size);

    // allocate recv buffer
    sham::DeviceBuffer<u8> recv_buf(comm_table.recv_total_size, dev_sched);

    // do the comm
    if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
        shamalgs::collective::sparse_exchange(dev_sched, send_buf, recv_buf, comm_table);
    } else {
        auto send_buf_host = send_buf.copy_to<sham::host>();
        auto recv_buf_host = recv_buf.copy_to<sham::host>();
        shamalgs::collective::sparse_exchange(dev_sched, send_buf_host, recv_buf_host, comm_table);
        recv_buf.copy_from(recv_buf_host);
    }

    // time to check

    size_t send_msg_idx = 0;
    size_t recv_msg_idx = 0;
    for (u32 i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].sender == shamcomm::world_rank()) {
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].message_size, test_elements[i].size);
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].rank_sender, test_elements[i].sender);
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].rank_receiver, test_elements[i].receiver);

            send_msg_idx++;
        }
        if (test_elements[i].receiver == shamcomm::world_rank()) {
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].message_size, test_elements[i].size);
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].rank_sender, test_elements[i].sender);
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].rank_receiver, test_elements[i].receiver);

            auto &ref_buf = all_bufs[i];
            sham::DeviceBuffer<u8> recov(test_elements[i].size, dev_sched);
            size_t begin = shambase::get_check_ref(
                comm_table.messages_recv[recv_msg_idx].message_bytebuf_offset_recv);
            size_t end = begin + test_elements[i].size;
            recv_buf.copy_range(begin, end, recov);

            logger::info_ln(
                "sparse exchange test",
                "rank :",
                shamcomm::world_rank(),
                "recv message : (",
                test_elements[i].sender,
                "->",
                test_elements[i].receiver,
                ") data :",
                recov.copy_to_stdvec());

            REQUIRE_EQUAL(recov.copy_to_stdvec(), ref_buf.copy_to_stdvec());

            recv_msg_idx++;
        }
        REQUIRE_EQUAL(comm_table.message_all[i].message_size, test_elements[i].size);
        REQUIRE_EQUAL(comm_table.message_all[i].rank_sender, test_elements[i].sender);
        REQUIRE_EQUAL(comm_table.message_all[i].rank_receiver, test_elements[i].receiver);
    }
}

TestStart(Unittest, "shamalgs/collective/test_sparse_exchange", testsparsexchg_2, -1) {

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "empty comm");
    }

    test_sparse_exchange({});

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "send to self");
    }

    {
        // everyone send to itself
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (i32 i = 0; i < shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{i, i, shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "send to next");
    }

    {
        // everyone send to next one
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (i32 i = 0; i < shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    i,
                    (i + 1) % shamcomm::world_size(),
                    shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "random test");
    }

    {
        // random test
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (u32 i = 0; i < 3 * shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements);
    }
}
