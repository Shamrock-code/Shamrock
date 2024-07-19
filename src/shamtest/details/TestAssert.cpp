// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestAssert.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
