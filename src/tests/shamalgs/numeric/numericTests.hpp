// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/details/random/random.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/random.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>

struct TestStreamCompact {

    using vFunctionCall = std::tuple<std::optional<sycl::buffer<u32>>, u32> (*)(
        sycl::queue &, sycl::buffer<u32> &, u32);

    vFunctionCall fct;

    explicit TestStreamCompact(vFunctionCall arg) : fct(arg) {};

    void check() {
        std::vector<u32> data{1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1};

        u32 len = data.size();

        auto buf = shamalgs::memory::vec_to_buf(data);

        auto [res, res_len] = fct(shamsys::instance::get_compute_queue(), buf, len);

        auto res_check = shamalgs::memory::buf_to_vec(*res, res_len);

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res_len, u32(idxs.size()));

        for (u32 idx = 0; idx < res_len; idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }
};

struct TestStreamCompactUSM {

    using vFunctionCall = sham::DeviceBuffer<u32> (*)(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &buf_flags, u32 len);

    vFunctionCall fct;

    explicit TestStreamCompactUSM(vFunctionCall arg) : fct(arg) {};

    void check_empty() {
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<u32> data{};

        u32 len = data.size();
        sham::DeviceBuffer<u32> buf(len, dev_sched);
        buf.copy_from_stdvec(data);

        sham::DeviceBuffer<u32> res = fct(dev_sched, buf, len);

        auto res_check = res.copy_to_stdvec();

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res.get_size(), u32(idxs.size()));

        for (u32 idx = 0; idx < res.get_size(); idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }

    void check_normal() {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<u32> data{1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1};

        u32 len = data.size();
        sham::DeviceBuffer<u32> buf(len, dev_sched);
        buf.copy_from_stdvec(data);

        sham::DeviceBuffer<u32> res = fct(dev_sched, buf, len);

        auto res_check = res.copy_to_stdvec();

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res.get_size(), u32(idxs.size()));

        for (u32 idx = 0; idx < res.get_size(); idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }

    void check_large() {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(u32(0x111), 10000, 0, 1);

        u32 len = data.size();
        sham::DeviceBuffer<u32> buf(len, dev_sched);
        buf.copy_from_stdvec(data);

        sham::DeviceBuffer<u32> res = fct(dev_sched, buf, len);

        auto res_check = res.copy_to_stdvec();

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res.get_size(), u32(idxs.size()));

        for (u32 idx = 0; idx < res.get_size(); idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }

    void check_normal_nores() {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<u32> data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        u32 len = data.size();
        sham::DeviceBuffer<u32> buf(len, dev_sched);
        buf.copy_from_stdvec(data);

        sham::DeviceBuffer<u32> res = fct(dev_sched, buf, len);

        auto res_check = res.copy_to_stdvec();

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res.get_size(), u32(idxs.size()));

        for (u32 idx = 0; idx < res.get_size(); idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }

    void check() {
        check_empty();
        check_normal_nores();
        check_normal();
        check_large();
    }
};
