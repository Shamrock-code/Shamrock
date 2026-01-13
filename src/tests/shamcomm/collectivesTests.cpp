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
