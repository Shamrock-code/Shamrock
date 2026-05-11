// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"
#include <unordered_map>
#include <array>
#include <random>

namespace {

    constexpr std::array<const char *, 10> word_pool = {{
        "cat",
        "dog",
        "bird",
        "fish",
        "wolf",
        "eagle",
        "tiger",
        "shark",
        "bear",
        "hawk",
    }};

    std::vector<std::vector<std::string>> build_histogram_dataset(i32 wsize) {
        std::mt19937 rng(42);
        std::vector<std::vector<std::string>> ref_base(wsize);
        std::uniform_int_distribution<i32> len_dist(1, 10);
        for (i32 r = 0; r < wsize; r++) {
            i32 n = len_dist(rng);
            ref_base[r].resize(n);
            for (i32 j = 0; j < n; j++) {
                std::uniform_int_distribution<i32> idx_dist(
                    0, static_cast<i32>(word_pool.size()) - 1);
                ref_base[r][j] = word_pool[idx_dist(rng)];
            }
        }
        return ref_base;
    }

    std::unordered_map<std::string, i32> compute_expected(i32 wsize) {
        std::unordered_map<std::string, i32> expected;
        auto ref_base = build_histogram_dataset(wsize);
        for (i32 i = 0; i < wsize; i++) {
            for (auto &w : ref_base[i])
                expected[w]++;
        }
        return expected;
    }

} // namespace

TestStart(Unittest, "shamcomm/collectives::gather_str", test_gather_str, 4) {

    std::array<std::string, 4> ref_base{
        "I'm a very important string",
        "But I'm a very important string",
        "Listen, I'm a very important string",
        "The most importantest string",
    };

    std::string result = "";
    if (shamcomm::world_rank() == 0) {
        for (u32 i = 0; i < ref_base.size(); i++) {
            result += ref_base[i];
        }
    }

    std::string send = ref_base[shamcomm::world_rank()];

    std::string recv = "random string"; // Just to check that it is overwritten

    shamcomm::gather_str(send, recv);

    REQUIRE_EQUAL(recv, result);
}

TestStart(Unittest, "shamcomm/collectives::allgather_str", test_allgather_str, 4) {

    std::array<std::string, 4> ref_base{
        "I'm a very important string",
        "But I'm a very important string",
        "Listen, I'm a very important string",
        "The most importantest string",
    };

    std::string result = "";
    for (u32 i = 0; i < ref_base.size(); i++) {
        result += ref_base[i];
    }

    std::string send = ref_base[shamcomm::world_rank()];

    std::string recv = "random string"; // Just to check that it is overwritten

    shamcomm::allgather_str(send, recv);

    REQUIRE_EQUAL(recv, result);
}

TestStart(Unittest, "shamcomm/collectives::string_histogram", test_string_histogram, -1) {

    i32 wsize     = shamcomm::world_size();
    auto ref_base = build_histogram_dataset(wsize);

    auto histogram = shamcomm::string_histogram(ref_base[shamcomm::world_rank()], ",");

    if (shamcomm::world_rank() == 0) {
        auto expected = compute_expected(wsize);
        for (auto &[word, cnt] : expected) {
            REQUIRE_EQUAL(histogram[word], cnt);
        }
    }
}

TestStart(Unittest, "shamcomm/collectives::all_string_histogram", test_all_string_histogram, -1) {

    i32 wsize     = shamcomm::world_size();
    auto ref_base = build_histogram_dataset(wsize);

    auto histogram = shamcomm::all_string_histogram(ref_base[shamcomm::world_rank()], ",");

    auto expected = compute_expected(wsize);
    for (auto &[word, cnt] : expected) {
        REQUIRE_EQUAL(histogram[word], cnt);
    }
}
