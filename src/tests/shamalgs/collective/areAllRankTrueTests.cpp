// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

    {
        // Test case 1: All ranks return true

        std::vector<bool> input = std::vector<bool>(world_size, true);
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL_NAMED("all_true", result, expected);
    }

    {
        // Test case 2: All ranks return false
        std::vector<bool> input = std::vector<bool>(world_size, false);
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL_NAMED("all_false", result, expected);
    }

    {
        // Test case 3: Mixed - some ranks true, some false (alternating pattern)
        std::vector<bool> input = std::vector<bool>(world_size);
        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            input[i] = (i % 2 == 0);
        }
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL_NAMED("mixed_alternating", result, expected);
    }

    {
        // Test case 4: Only rank 0 returns false, others true
        std::vector<bool> input = std::vector<bool>(world_size);
        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            input[i] = (i != 0);
        }
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL_NAMED("rank0_false", result, expected);
    }

    {
        // Test case 5: Only last rank returns false, others true
        std::vector<bool> input = std::vector<bool>(world_size);
        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            input[i] = (i != shamcomm::world_size() - 1);
        }
        bool result   = shamalgs::collective::are_all_rank_true(input[world_rank], MPI_COMM_WORLD);
        bool expected = reference_are_all_rank_true(input);
        REQUIRE_EQUAL_NAMED("last_rank_false", result, expected);
    }
}
