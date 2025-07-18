// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TestResult.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "TestResult.hpp"
#include "shambase/bytestream.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestAssertList.hpp"
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace shamtest::details {

    std::string TestResult::serialize_json() {

        using namespace shamsys::instance;

        auto get_type_name = [](TestType t) -> std::string {
            switch (t) {
            case Benchmark: return "Benchmark";
            case LongBenchmark: return "LongBenchmark";
            case ValidationTest: return "ValidationTest";
            case LongValidationTest: return "LongValidationTest";
            case Unittest: return "Unittest";
            }
        };

        auto get_str = [&]() -> std::string {
            return "{\n"
                   R"(    "type" : ")"
                   + get_type_name(type) + "\",\n" + R"(    "name" : ")" + name + "\",\n"
                   + R"(    "compute_queue" : ")"
                   + get_compute_queue().get_device().get_info<sycl::info::device::name>() + "\",\n"
                   + R"(    "alt_queue" : ")"
                   + get_alt_queue().get_device().get_info<sycl::info::device::name>() + "\",\n"
                   + R"(    "world_rank" : )" + std::to_string(world_rank) + ",\n"
                   + R"(    "asserts" : )" + shambase::increase_indent(asserts.serialize_json())
                   + ",\n" + R"(    "test_data" : )"
                   + shambase::increase_indent(test_data.serialize_json()) + "\n" + "}";
        };

        return get_str();
    }

    void TestResult::serialize(std::basic_stringstream<byte> &stream) {

        shamlog_debug_mpi_ln("TEST", "serialize :", name);

        shambase::stream_write(stream, type);

        shambase::stream_write_string(stream, name);

        shambase::stream_write(stream, world_rank);

        asserts.serialize(stream);
        test_data.serialize(stream);

        shambase::stream_write_string(stream, tex_output);
    }

    TestResult TestResult::deserialize(std::basic_stringstream<byte> &reader) {
        TestType type;
        std::string name;
        u32 world_rank;
        std::string tex_output;

        shambase::stream_read(reader, type);

        shambase::stream_read_string(reader, name);
        shamlog_debug_mpi_ln("TEST", "deserialize :", name);

        shambase::stream_read(reader, world_rank);

        TestAssertList asserts = TestAssertList::deserialize(reader);
        TestDataList test_data = TestDataList::deserialize(reader);

        shambase::stream_read_string(reader, tex_output);

        return TestResult{
            type,
            name,
            world_rank,
            std::move(asserts),
            std::move(test_data),
            std::move(tex_output)};
    }

} // namespace shamtest::details
