// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/collective/are_all_rank_true", test_are_all_rank_true, -1) {

    u32 world_size = shamcomm::world_size();
    u32 world_rank = shamcomm::world_rank();

    auto reference_are_all_rank_true = [](std::vector<bool> input) {
        bool out = true;
        for (bool tmp : input) {
            out = out && tmp;
        }
        return out;
    };

    auto run_test = [&](const auto &input_generator) {
        std::vector<bool> input(world_size);
        for (u32 i = 0; i < world_size; ++i) {
            input[i] = input_generator(i);
        }
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL(result, expected);
    };

    // Test case 1: All ranks return true
    run_test([](u32) {
        return true;
    });

    // Test case 2: All ranks return false
    run_test([](u32) {
        return false;
    });

    // Test case 3: Mixed - some ranks true, some false (alternating pattern)
    run_test([](u32 i) {
        return (i % 2 == 0);
    });

    // Test case 4: Only rank 0 returns false, others true
    run_test([](u32 i) {
        return (i != 0);
    });

    // Test case 5: Only last rank returns false, others true
    run_test([world_size](u32 i) {
        return (i != world_size - 1);
    });
}
