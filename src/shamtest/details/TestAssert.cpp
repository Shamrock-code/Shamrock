// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestAssert.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "TestAssert.hpp"
#include "shambase/bytestream.hpp"

namespace shamtest::details {

    std::string TestAssert::serialize_json() {
        std::string acc = "\n{\n";

        acc += R"(    "value" : )" + std::to_string(value) + ",\n";
        acc += R"(    "name" : ")" + name + "\"";

        if (!comment.empty()) {
            acc += ",\n"
                   R"(    "comment" : ")"
                   + comment + "\"";
        }

        acc += "\n}";
        return acc;
    }

    void TestAssert::serialize(std::basic_stringstream<byte> &stream) {
        byte bl = value;
        shambase::stream_write(stream, bl);
        shambase::stream_write_string(stream, name);
        shambase::stream_write_string(stream, comment);
    }

    TestAssert TestAssert::deserialize(std::basic_stringstream<byte> &stream) {
        byte bl;
        std::string name;
        std::string comment;
        shambase::stream_read(stream, bl);
        shambase::stream_read_string(stream, name);
        shambase::stream_read_string(stream, comment);

        return {bool(bl), name, comment};
    }

} // namespace shamtest::details
