// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestAssertList.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "TestAssertList.hpp"
#include "shambase/bytestream.hpp"
#include "shamtest/details/TestAssert.hpp"
#include <sstream>
#include <vector>

namespace shamtest::details {

    std::string TestAssertList::serialize_json() {
        std::string acc = "\n[\n";

        for (u32 i = 0; i < asserts.size(); i++) {
            acc += shambase::increase_indent(asserts[i].serialize_json());
            if (i < asserts.size() - 1) {
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }

    void TestAssertList::serialize(std::basic_stringstream<byte> &stream) {
        shambase::stream_write_vector(stream, asserts);
    }

    TestAssertList TestAssertList::deserialize(std::basic_stringstream<byte> &stream) {
        std::vector<TestAssert> tmp;
        shambase::stream_read_vector(stream, tmp);
        return {std::move(tmp)};
    }

} // namespace shamtest::details
