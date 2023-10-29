// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/collective/indexing.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <array>

TestStart(Unittest, "shamalgs/collective/indexing/fetch_view", test_collective_fetch_view, 2){
    
    std::array cnts = {
        10_u64,
        14_u64
    };

    using namespace shamalgs::collective;
    using namespace shamsys::instance;

    ViewInfo ret =  fetch_view(cnts[shamcomm::world_rank()]);

    std::array excpected_offsets = {
        0_u64,
        10_u64
    };

    shamtest::asserts().assert_equal("offset", ret.head_offset, excpected_offsets[shamcomm::world_rank()]);
    shamtest::asserts().assert_equal("sum", ret.total_byte_count, 24_u64);

}