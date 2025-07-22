// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestData.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "TestData.hpp"
#include "shambase/bytestream.hpp"
#include "shambase/string.hpp"

namespace shamtest::details {

    std::string TestData::serialize_json() {
        std::string acc = "\n{\n";

        acc += R"(    "dataset_name" : ")" + dataset_name + "\",\n";
        acc += R"(    "dataset" : )"
               "\n    [\n";

        for (u32 i = 0; i < dataset.size(); i++) {
            acc += shambase::increase_indent(dataset[i].serialize_json());
            if (i < dataset.size() - 1) {
                acc += ",";
            }
        }

        acc += "]";

        acc += "\n}";
        return acc;
    }

    void TestData::serialize(std::basic_stringstream<byte> &stream) {
        shambase::stream_write_string(stream, dataset_name);
        shambase::stream_write_vector(stream, dataset);
    }

    TestData TestData::deserialize(std::basic_stringstream<byte> &stream) {

        TestData out;

        shambase::stream_read_string(stream, out.dataset_name);
        shambase::stream_read_vector(stream, out.dataset);

        return out;
    }

} // namespace shamtest::details
