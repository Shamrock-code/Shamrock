// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestDataList.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "TestDataList.hpp"
#include "shambase/bytestream.hpp"
#include "shambase/string.hpp"
#include <sstream>

namespace shamtest::details {

    std::string TestDataList::serialize_json() {
        std::string acc = "\n[\n";

        for (u32 i = 0; i < test_data.size(); i++) {
            acc += shambase::increase_indent(test_data[i].serialize_json());
            if (i < test_data.size() - 1) {
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }

    void TestDataList::serialize(std::basic_stringstream<byte> &stream) {
        shambase::stream_write_vector(stream, test_data);
    }
    TestDataList TestDataList::deserialize(std::basic_stringstream<byte> &stream) {
        TestDataList out;

        shambase::stream_read_vector(stream, out.test_data);

        return {};
    }

} // namespace shamtest::details