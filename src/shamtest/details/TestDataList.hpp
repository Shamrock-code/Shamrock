// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TestDataList.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "TestData.hpp"
#include <vector>

namespace shamtest::details {

    struct TestDataList {
        std::vector<TestData> test_data;

        // define member function here
        // to register test data

        [[nodiscard]] inline TestData &new_dataset(std::string name) {
            test_data.push_back(TestData{std::move(name), {}});
            return test_data.back();
        }

        std::string serialize_json();
        void serialize(std::basic_stringstream<byte> &stream);
        static TestDataList deserialize(std::basic_stringstream<byte> &reader);
    };

}