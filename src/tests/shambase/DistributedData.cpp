// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/DistributedData::add_obj", distributedDatatests_add_obj, 1) {
    using namespace shambase;

    {
        DistributedData<int> data{};
        auto it = data.add_obj(1, 42);
        REQUIRE(it->first == 1);
        REQUIRE(it->second == 42);
        REQUIRE(data.get_element_count() == 1);
    }

    {
        DistributedData<int> data{};
        data.add_obj(1, 42);
        REQUIRE_EXCEPTION_THROW(data.add_obj(1, 43), std::runtime_error);
        REQUIRE(data.get_element_count() == 1);
    }
}
