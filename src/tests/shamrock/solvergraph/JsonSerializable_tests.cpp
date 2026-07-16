// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/solvergraph/JsonSerializable.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <string>

class TestClassSerialization : public shamrock::solvergraph::JsonSerializable {
    public:
    int value;
    std::string name;

    TestClassSerialization(int value, std::string name) : value(value), name(std::move(name)) {}

    void to_json(nlohmann::json &j) override {
        j["value"] = value;
        j["name"]  = name;
    }

    static TestClassSerialization from_json(const nlohmann::json &j) {
        return TestClassSerialization(j.at("value").get<int>(), j.at("name").get<std::string>());
    }

    std::string type_name() override { return "TestClassSerialization"; }
};

NEW_TEST(Unittest, "shamrock/solvergraph/JsonSerializable", 1) {
    using namespace shamrock::solvergraph;

    JsonSerializable_registry::instance().registerType<TestClassSerialization>(
        "TestClassSerialization");

    {
        TestClassSerialization original{42, "hello"};

        nlohmann::json j;
        original.to_json(j);
        j["type"] = original.type_name();

        auto ptr       = JsonSerializable::from_json(j);
        auto *restored = dynamic_cast<TestClassSerialization *>(ptr.get());

        REQUIRE(restored != nullptr);
        REQUIRE_EQUAL(restored->value, 42);
        REQUIRE_EQUAL(restored->name, "hello");
        REQUIRE_EQUAL(restored->type_name(), "TestClassSerialization");
    }

    {
        nlohmann::json j;
        j["type"] = "NonExistentType";

        REQUIRE_EXCEPTION_THROW(JsonSerializable::from_json(j), std::runtime_error);
    }
}
