// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/indexing.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <array>

TestStart(Unittest, "shamalgs/collective/indexing/fetch_view", test_collective_fetch_view, 2) {

    std::array cnts = {10_u64, 14_u64};

    using namespace shamalgs::collective;
    using namespace shamsys::instance;

    ViewInfo ret = fetch_view(cnts[shamcomm::world_rank()]);

    std::array excpected_offsets = {0_u64, 10_u64};

    REQUIRE_EQUAL_NAMED("offset", ret.head_offset, excpected_offsets[shamcomm::world_rank()]);
    REQUIRE_EQUAL_NAMED("sum", ret.total_byte_count, 24_u64);
}
