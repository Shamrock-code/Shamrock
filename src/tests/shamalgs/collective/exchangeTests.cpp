// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_int.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"
#include <array>
#include <random>
#include <vector>

TestStart(Unittest, "shamalgs/collective/exchange/vector_allgatherv", test_vector_allgatherv, -1) {

    {
        // Test case 1: Random sized vectors
        std::mt19937 eng(0x1111);

        std::vector<std::vector<u32>> test_array(shamcomm::world_size());
        std::vector<u32> ref_vec;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            auto &vec = test_array[i];
            u32 random_size
                = shamalgs::primitives::mock_value<u32>(eng, 0, 10); // Random size between 1 and 10
            vec.resize(random_size);
            for (auto &num : vec) {
                num = shamalgs::primitives::mock_value<u32>(
                    eng, 1, 100000); // Random number between 0 and 100
                ref_vec.push_back(num);
            }
        }

        auto &source_vec = test_array[shamcomm::world_rank()];

        std::vector<u32> recv;
        shamalgs::collective::vector_allgatherv(source_vec, recv, MPI_COMM_WORLD);

        REQUIRE_EQUAL(ref_vec, recv);
    }

    {
        // Test case 2: Some ranks have empty vectors (alternating pattern)
        std::mt19937 eng(0x2222);

        std::vector<std::vector<u32>> test_array(shamcomm::world_size());
        std::vector<u32> ref_vec;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            auto &vec = test_array[i];
            // Make every other rank empty
            u32 random_size = (i % 2 == 0) ? 0 : shamalgs::primitives::mock_value<u32>(eng, 1, 8);
            vec.resize(random_size);
            for (auto &num : vec) {
                num = shamalgs::primitives::mock_value<u32>(eng, 1, 100000);
                ref_vec.push_back(num);
            }
        }

        auto &source_vec = test_array[shamcomm::world_rank()];

        std::vector<u32> recv;
        shamalgs::collective::vector_allgatherv(source_vec, recv, MPI_COMM_WORLD);

        REQUIRE_EQUAL(ref_vec, recv);
    }

    {
        // Test case 3: All ranks have empty vectors
        std::vector<std::vector<u32>> test_array(shamcomm::world_size());
        std::vector<u32> ref_vec;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            auto &vec = test_array[i];
            vec.resize(0); // All vectors are empty
        }

        auto &source_vec = test_array[shamcomm::world_rank()];

        std::vector<u32> recv;
        shamalgs::collective::vector_allgatherv(source_vec, recv, MPI_COMM_WORLD);

        REQUIRE_EQUAL(ref_vec, recv);
    }
}

TestStart(
    Unittest,
    "shamalgs/collective/exchange/vector_allgatherv_large",
    test_vector_allgatherv_large,
    -1) {

    {
        // Test case 1: Random sized vectors
        std::mt19937 eng(0x1111);

        std::vector<std::vector<u32>> test_array(shamcomm::world_size());
        std::vector<u32> ref_vec;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            auto &vec       = test_array[i];
            u32 random_size = shamalgs::primitives::mock_value<u32>(
                eng, 0, 200); // Random size between 1 and 10
            vec.resize(random_size);
            for (auto &num : vec) {
                num = shamalgs::primitives::mock_value<u32>(
                    eng, 1, 100000); // Random number between 0 and 100
                ref_vec.push_back(num);
            }
        }

        auto &source_vec = test_array[shamcomm::world_rank()];

        std::vector<u32> recv;
        shamalgs::collective::vector_allgatherv_large(
            source_vec, get_mpi_type<u32>(), recv, get_mpi_type<u32>(), MPI_COMM_WORLD, 10);

        REQUIRE_EQUAL(ref_vec, recv);
    }

    {
        // Test case 3: All ranks have empty vectors
        std::vector<std::vector<u32>> test_array(shamcomm::world_size());
        std::vector<u32> ref_vec;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            auto &vec = test_array[i];
            vec.resize(0); // All vectors are empty
        }

        auto &source_vec = test_array[shamcomm::world_rank()];

        std::vector<u32> recv;
        shamalgs::collective::vector_allgatherv_large(
            source_vec, get_mpi_type<u32>(), recv, get_mpi_type<u32>(), MPI_COMM_WORLD, 5);

        REQUIRE_EQUAL(ref_vec, recv);
    }
}
