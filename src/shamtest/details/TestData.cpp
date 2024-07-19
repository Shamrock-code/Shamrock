// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestData.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
