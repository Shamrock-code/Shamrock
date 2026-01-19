// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DataNode.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "DataNode.hpp"
#include "shambase/bytestream.hpp"
#include "shambase/string.hpp"

namespace shamtest::details {

    std::string serialize_vec(const std::vector<f64> &vec) {
        std::string acc = "\n[\n";

        for (u32 i = 0; i < vec.size(); i++) {
            acc += shambase::format_printf("%e", vec[i]);
            if (i < vec.size() - 1) {
                acc += ", ";
            }
        }

        acc += "\n]";
        return acc;
    }

    std::string DataNode::serialize_json() {
        std::string acc = "\n{\n";

        acc += R"(    "name" : ")" + name + "\",\n";

        acc += R"(    "data" : )"
               "\n"
               + serialize_vec(data) + "\n";

        acc += "\n}";
        return acc;
    }

    void DataNode::serialize(std::basic_stringstream<byte> &stream) {
        shambase::stream_write_string(stream, name);
        shambase::stream_write_vector_trivial(stream, data);
    }
    DataNode DataNode::deserialize(std::basic_stringstream<byte> &stream) {

        DataNode out{};

        shambase::stream_read_string(stream, out.name);
        shambase::stream_read_vector_trivial(stream, out.data);
        return out;
    }
} // namespace shamtest::details
