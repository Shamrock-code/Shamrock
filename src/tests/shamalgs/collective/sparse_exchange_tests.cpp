// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/sequentialize.hpp"
#include "shamalgs/collective/sparse_exchange.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <fmt/std.h>
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

#if false
void validate_comm_table(
    const std::vector<TestElement> &test_elements,
    const shamalgs::collective::CommTable &comm_table,
    size_t max_alloc_size) {

    std::vector<shamalgs::collective::CommMessageInfo> messages_send;

    std::vector<size_t> total_send_sizes = {0};
    std::vector<size_t> total_recv_sizes = {0};
    shamalgs::collective::sequentialize([&]() {
        u32 send_buf_id    = 0;
        u32 recv_buf_id    = 0;
        size_t send_offset = 0;
        size_t recv_offset = 0;
        for (u32 i = 0; i < test_elements.size(); i++) {
            if (test_elements[i].sender == shamcomm::world_rank()) {
                messages_send.push_back(
                    shamalgs::collective::CommMessageInfo{
                        test_elements[i].size,
                        test_elements[i].sender,
                        test_elements[i].receiver,
                        std::nullopt,
                        std::nullopt,
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
                    ")");

                if (send_offset + test_elements[i].size > max_alloc_size) {
                    send_buf_id++;
                    send_offset = 0;
                    total_send_sizes.push_back(0);
                }

                total_send_sizes.at(send_buf_id) += test_elements[i].size;
            }
            if (test_elements[i].receiver == shamcomm::world_rank()) {
                if (recv_offset + test_elements[i].size > max_alloc_size) {
                    recv_buf_id++;
                    recv_offset = 0;
                    total_recv_sizes.push_back(0);
                }

                total_recv_sizes.at(recv_buf_id) += test_elements[i].size;
            }
        }
    });

    REQUIRE_EQUAL(comm_table.send_total_sizes, total_send_sizes);
    REQUIRE_EQUAL(comm_table.recv_total_sizes, total_recv_sizes);

    shamalgs::collective::sequentialize([&]() {
        size_t send_msg_idx = 0;
        size_t recv_msg_idx = 0;
        for (u32 i = 0; i < test_elements.size(); i++) {
            if (test_elements[i].sender == shamcomm::world_rank()) {
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].message_size, test_elements[i].size);
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].rank_sender, test_elements[i].sender);
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].rank_receiver,
                    test_elements[i].receiver);

                send_msg_idx++;
            }
            if (test_elements[i].receiver == shamcomm::world_rank()) {
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].message_size, test_elements[i].size);
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].rank_sender, test_elements[i].sender);
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].rank_receiver,
                    test_elements[i].receiver);

                auto &ref_buf = all_bufs[i];
                sham::DeviceBuffer<u8> recov(test_elements[i].size, dev_sched);
                auto off_info = shambase::get_check_ref(
                    comm_table.messages_recv[recv_msg_idx].message_bytebuf_offset_recv);
                size_t begin = off_info.data_offset;
                size_t end   = begin + test_elements[i].size;
                shambase::get_check_ref(recv_bufs.at(off_info.buf_id))
                    .copy_range(begin, end, recov);

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
    });
}
#endif

template<>
struct fmt::formatter<shamalgs::collective::CommMessageBufOffset> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(shamalgs::collective::CommMessageBufOffset c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(), "Offset(buf_id : {}, data_offset : {})", c.buf_id, c.data_offset);
    }
};

template<>
struct fmt::formatter<shamalgs::collective::CommMessageInfo> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(shamalgs::collective::CommMessageInfo c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(),
            "Info(size : {}, sender : {}, receiver : {}, offset_send : {}, offset_recv : {})",
            c.message_size,
            c.rank_sender,
            c.rank_receiver,
            c.message_bytebuf_offset_send,
            c.message_bytebuf_offset_recv);
    }
};

void print_comm_table(const shamalgs::collective::CommTable &comm_table) {
    std::stringstream ss;
    ss << shambase::format(
        "messages_send : [\n    {}\n]\n", fmt::join(comm_table.messages_send, "\n    "));
    ss << shambase::format(
        "messages_recv : [\n    {}\n]\n", fmt::join(comm_table.messages_recv, "\n    "));
    ss << shambase::format(
        "message_all : [\n    {}\n]\n", fmt::join(comm_table.message_all, "\n    "));
    ss << shambase::format("send_message_global_ids : {}\n", comm_table.send_message_global_ids);
    ss << shambase::format("recv_message_global_ids : {}\n", comm_table.recv_message_global_ids);
    ss << shambase::format("send_total_sizes : {}\n", comm_table.send_total_sizes);
    ss << shambase::format("recv_total_sizes : {}\n", comm_table.recv_total_sizes);
    logger::info_ln(
        "sparse exchange test", "rank :", shamcomm::world_rank(), "comm table :", "\n" + ss.str());
}

void test_sparse_exchange(std::vector<TestElement> test_elements, size_t max_alloc_size) {
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    reorder_msg(test_elements);

    std::vector<sham::DeviceBuffer<u8>> all_bufs;

    std::mt19937 eng(0x123);
    for (const auto &test_element : test_elements) {
        all_bufs.push_back(
            shamalgs::random::mock_buffer_usm<u8>(dev_sched, eng(), test_element.size));
    }

    std::vector<shamalgs::collective::CommMessageInfo> messages_send;

    for (u32 i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].sender == shamcomm::world_rank()) {
            messages_send.push_back(
                shamalgs::collective::CommMessageInfo{
                    test_elements[i].size,
                    test_elements[i].sender,
                    test_elements[i].receiver,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                });
        }
    }

    shamalgs::collective::CommTable comm_table
        = shamalgs::collective::build_sparse_exchange_table(messages_send, max_alloc_size);

    // print_comm_table(comm_table);

    // allocate send bufs
    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> send_bufs{};

    for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
        send_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.send_total_sizes[i], dev_sched));
    }

    // push data to the comm buf
    for (size_t i = 0; i < comm_table.messages_send.size(); i++) {
        auto msg_info        = comm_table.messages_send[i];
        size_t global_msg_id = comm_table.send_message_global_ids[i];

        auto off_info = shambase::get_check_ref(msg_info.message_bytebuf_offset_send);

        auto &source = all_bufs.at(global_msg_id);
        auto &dest   = shambase::get_check_ref(send_bufs.at(off_info.buf_id));

        source.copy_range_offset(0, source.get_size(), dest, off_info.data_offset);
    }

    // allocate recv bufs
    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> recv_bufs{};

    for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
        recv_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.recv_total_sizes[i], dev_sched));
    }

    // do the comm
    if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
        shamalgs::collective::sparse_exchange(dev_sched, send_bufs, recv_bufs, comm_table);
    } else {
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> send_bufs_host{};
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> recv_bufs_host{};

        for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
            send_bufs_host.push_back(
                std::make_unique<sham::DeviceBuffer<u8, sham::host>>(
                    send_bufs[i]->copy_to<sham::host>()));
        }
        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            recv_bufs_host.push_back(
                std::make_unique<sham::DeviceBuffer<u8, sham::host>>(
                    comm_table.recv_total_sizes[i], dev_sched));
        }

        shamalgs::collective::sparse_exchange(
            dev_sched, send_bufs_host, recv_bufs_host, comm_table);
        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            recv_bufs[i]->copy_from(*recv_bufs_host[i]);
        }
    }

    {
        std::stringstream ss;
        ss << "send bufs :\n";
        for (size_t i = 0; i < send_bufs.size(); i++) {
            ss << "buf " << i << " : " << shambase::format("{}", send_bufs[i]->copy_to_stdvec())
               << "\n";
        }
        ss << "recv bufs :\n";
        for (size_t i = 0; i < recv_bufs.size(); i++) {
            ss << "buf " << i << " : " << shambase::format("{}", recv_bufs[i]->copy_to_stdvec())
               << "\n";
        }
        logger::info_ln("sparse exchange test", "rank :", shamcomm::world_rank(), ss.str());
    }

    // time to check
    std::vector<sham::DeviceBuffer<u8>> recv_messages;

    for (size_t i = 0; i < comm_table.messages_recv.size(); i++) {
        auto msg_info        = comm_table.messages_recv[i];
        size_t global_msg_id = comm_table.recv_message_global_ids[i];

        auto off_info
            = shambase::get_check_ref(comm_table.messages_recv[i].message_bytebuf_offset_recv);

        sham::DeviceBuffer<u8> recov(test_elements[global_msg_id].size, dev_sched);

        size_t begin = off_info.data_offset;
        size_t end   = begin + test_elements[global_msg_id].size;
        shambase::get_check_ref(recv_bufs.at(off_info.buf_id)).copy_range(begin, end, recov);
        recv_messages.push_back(std::move(recov));
    }

    // validate
    u32 recv_idx = 0;
    for (size_t i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].receiver == shamcomm::world_rank()) {
            REQUIRE_EQUAL(recv_messages[recv_idx].copy_to_stdvec(), all_bufs[i].copy_to_stdvec());
            logger::info_ln(
                "sparse exchange test",
                "rank :",
                shamcomm::world_rank(),
                "recv message : (",
                test_elements[i].sender,
                "->",
                test_elements[i].receiver,
                ") data :",
                recv_messages[recv_idx].copy_to_stdvec(),
                "data ref :",
                all_bufs[i].copy_to_stdvec(),
                "valid :",
                recv_messages[recv_idx].copy_to_stdvec() == all_bufs[i].copy_to_stdvec());
            recv_idx++;
        }
    }
}

TestStart(Unittest, "shamalgs/collective/test_sparse_exchange", testsparsexchg_2, -1) {

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "empty comm");
    }

    test_sparse_exchange({}, i32_max);

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
        test_sparse_exchange(test_elements, i32_max);
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
        test_sparse_exchange(test_elements, i32_max);
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
        test_sparse_exchange(test_elements, i32_max);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "random test (force multiple bufs)");
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
        test_sparse_exchange(test_elements, 20);
    }
}
