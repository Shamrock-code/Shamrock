// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
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

TestStart(Unittest, "shamalgs/collective/exchange/vector_allgatherv", test_vector_allgatherv, 4) {

    std::mt19937 eng(0x1111);

    std::array<std::vector<u32>, 4> test_array;
    std::vector<u32> ref_vec;

    for (auto &vec : test_array) {
        u32 random_size
            = shamalgs::random::mock_value<u32>(eng, 1, 10); // Random size between 1 and 10
        vec.resize(random_size);
        for (auto &num : vec) {
            num = shamalgs::random::mock_value<u32>(
                eng, 1, 100000); // Random number between 0 and 100
            ref_vec.push_back(num);
        }
    }

    auto &source_vec = test_array[shamcomm::world_rank()];

    std::vector<u32> recv;
    shamalgs::collective::vector_allgatherv(source_vec, recv, MPI_COMM_WORLD);

    REQUIRE_EQUAL(ref_vec, recv);
}
