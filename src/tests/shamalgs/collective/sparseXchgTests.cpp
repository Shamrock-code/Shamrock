// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/sparseXchg.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

void sparse_comm_test(std::string prefix, std::shared_ptr<sham::DeviceScheduler> qdet) {

    using namespace shamalgs::collective;
    using namespace shamsys::instance;
    using namespace shamsys;
    using namespace shamcomm;

    const i32 wsize = world_size();
    const i32 wrank = world_rank();

    u32 num_buf        = wsize * 5;
    u32 nbytes_per_buf = 1e4;

    u64 seed = 0x123;
    std::mt19937 eng(seed);

    struct RefBuff {
        i32 sender_rank;
        i32 receiver_rank;
        std::unique_ptr<sycl::buffer<u8>> payload;
    };

    struct TestElements {
        std::vector<RefBuff> elements;

        void add_element(std::mt19937 &eng, u32 wsize, u64 bytes) {
            u64 rnd = eng();
            elements.push_back(RefBuff{
                shamalgs::mock_value<i32>(eng, 0, wsize - 1),
                shamalgs::mock_value<i32>(eng, 0, wsize - 1),
                std::make_unique<sycl::buffer<u8>>(shamalgs::random::mock_buffer<u8>(
                    rnd, shamalgs::mock_value<i32>(eng, 1, bytes)))});
        }

        void sort_input() {
            std::sort(elements.begin(), elements.end(), [](const auto &lhs, const auto &rhs) {
                return lhs.sender_rank < rhs.sender_rank;
            });
        }
    };

    TestElements tests;
    for (u32 i = 0; i < num_buf; i++) {
        tests.add_element(eng, wsize, nbytes_per_buf);
    }
    tests.sort_input();

    // make comm bufs

    std::vector<SendPayload> sendop;

    u32 idx = 0;
    for (RefBuff &bufinfo : tests.elements) {
        if (bufinfo.sender_rank == world_rank()) {
            sendop.push_back(SendPayload{
                bufinfo.receiver_rank,
                std::make_unique<CommunicationBuffer>(*bufinfo.payload, qdet)});

            REQUIRE_EQUAL(sendop[idx].payload->get_size(), bufinfo.payload->size());

            idx++;
        }
    }

    std::vector<RecvPayload> recvop;
    base_sparse_comm(get_compute_scheduler_ptr(), sendop, recvop);

    std::vector<RefBuff> recv_data;
    for (RecvPayload &load : recvop) {
        recv_data.push_back(RefBuff{
            load.sender_ranks,
            wrank,
            std::make_unique<sycl::buffer<u8>>(load.payload->copy_back())});
    }

    logger::raw_ln("ref data : ");
    for (RefBuff &ref : tests.elements) {
        logger::raw_ln(shambase::format(
            "[{:2}] {} -> {} ({})",
            wrank,
            ref.sender_rank,
            ref.receiver_rank,
            ref.payload->size()));
    }

    logger::raw_ln("recv data : ");
    for (RefBuff &ref : recv_data) {
        logger::raw_ln(shambase::format(
            "[{:2}] {} -> {} ({})",
            wrank,
            ref.sender_rank,
            ref.receiver_rank,
            ref.payload->size()));
    }

    u32 ref_idx = 0;
    for (RefBuff &ref : tests.elements) {
        if (ref.receiver_rank == wrank) {

            if (ref_idx < recv_data.size()) {

                RefBuff &recv_buf = recv_data[ref_idx];

                REQUIRE_EQUAL_NAMED(prefix + "same sender", recv_buf.sender_rank, ref.sender_rank);
                REQUIRE_EQUAL_NAMED(
                    prefix + "same receiver", recv_buf.receiver_rank, ref.receiver_rank);

                REQUIRE_EQUAL_NAMED(
                    prefix + "same buf size",
                    ref.payload->get_size(),
                    recv_buf.payload->get_size());
                REQUIRE_NAMED(
                    prefix + "same buffer",
                    shamalgs::reduction::equals_ptr(
                        get_compute_queue(), ref.payload, recv_buf.payload));

            } else {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    prefix + "missing recv mesages");
            }

            ref_idx++;
        }
    }
}

TestStart(Unittest, "shamalgs/collective/sparseXchg", testsparsexchg, -1) {

    sparse_comm_test("", shamsys::instance::get_compute_scheduler_ptr());
}
