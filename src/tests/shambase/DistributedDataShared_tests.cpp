// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedDataShared.hpp"
#include "shamtest/shamtest.hpp"
#include <string>
#include <vector>

TestStart(
    Unittest, "shambase/DistributedDataShared::add_obj", distributedDataSharedTests_add_obj, 1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};
        data.add_obj(1, 2, 42);
        REQUIRE(data.get_element_count() == 1);
        REQUIRE(data.has_key(1, 2));
    }

    {
        DistributedDataShared<int> data{};
        data.add_obj(1, 2, 42);
        data.add_obj(1, 2, 43); // Multiple objects for same patch pair should be allowed
        REQUIRE(data.get_element_count() == 2);
        REQUIRE(data.has_key(1, 2));
    }

    {
        DistributedDataShared<std::string> data{};
        std::string test_value = "test_string";
        data.add_obj(5, 10, std::move(test_value));
        REQUIRE(data.get_element_count() == 1);
        REQUIRE(data.has_key(5, 10));
    }
}

TestStart(
    Unittest, "shambase/DistributedDataShared::for_each", distributedDataSharedTests_for_each, 1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};
        data.add_obj(1, 2, 42);
        data.add_obj(3, 4, 84);
        data.add_obj(1, 2, 21); // Multiple objects for same patch pair

        std::vector<std::tuple<u64, u64, int>> collected;
        data.for_each([&](u64 left, u64 right, int &value) {
            collected.emplace_back(left, right, value);
        });

        REQUIRE(collected.size() == 3);

        // Check that all expected values are present
        bool found_42 = false, found_84 = false, found_21 = false;
        for (const auto &[left, right, val] : collected) {
            if (left == 1 && right == 2 && val == 42)
                found_42 = true;
            if (left == 3 && right == 4 && val == 84)
                found_84 = true;
            if (left == 1 && right == 2 && val == 21)
                found_21 = true;
        }
        REQUIRE(found_42);
        REQUIRE(found_84);
        REQUIRE(found_21);
    }

    {
        DistributedDataShared<int> data{};
        data.add_obj(1, 2, 10);

        // Test modification through for_each
        data.for_each([](u64 left, u64 right, int &value) {
            value *= 2;
        });

        bool found_modified = false;
        data.for_each([&](u64 left, u64 right, int &value) {
            if (left == 1 && right == 2 && value == 20) {
                found_modified = true;
            }
        });
        REQUIRE(found_modified);
    }
}

TestStart(
    Unittest,
    "shambase/DistributedDataShared::tranfer_all",
    distributedDataSharedTests_transfer_all,
    1) {
    using namespace shambase;

    {
        DistributedDataShared<int> source{};
        DistributedDataShared<int> destination{};

        source.add_obj(1, 2, 42);
        source.add_obj(3, 4, 84);
        source.add_obj(5, 6, 21);
        source.add_obj(7, 8, 63);

        REQUIRE(source.get_element_count() == 4);
        REQUIRE(destination.get_element_count() == 0);

        // Transfer objects where left_id > 3
        source.tranfer_all(
            [](u64 left, u64 right) {
                return left > 3;
            },
            destination);

        REQUIRE(source.get_element_count() == 2);
        REQUIRE(destination.get_element_count() == 2);

        // Check source still has the correct objects
        REQUIRE(source.has_key(1, 2));
        REQUIRE(source.has_key(3, 4));
        REQUIRE(!source.has_key(5, 6));
        REQUIRE(!source.has_key(7, 8));

        // Check destination has the transferred objects
        REQUIRE(!destination.has_key(1, 2));
        REQUIRE(!destination.has_key(3, 4));
        REQUIRE(destination.has_key(5, 6));
        REQUIRE(destination.has_key(7, 8));
    }

    {
        DistributedDataShared<int> source{};
        DistributedDataShared<int> destination{};

        source.add_obj(1, 1, 10);
        source.add_obj(2, 2, 20);

        // Transfer objects where left_id == right_id
        source.tranfer_all(
            [](u64 left, u64 right) {
                return left == right;
            },
            destination);

        REQUIRE(source.get_element_count() == 0);
        REQUIRE(destination.get_element_count() == 2);
    }
}

TestStart(
    Unittest, "shambase/DistributedDataShared::has_key", distributedDataSharedTests_has_key, 1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};

        REQUIRE(!data.has_key(1, 2));

        data.add_obj(1, 2, 42);
        REQUIRE(data.has_key(1, 2));
        REQUIRE(!data.has_key(2, 1)); // Order matters
        REQUIRE(!data.has_key(1, 3));
        REQUIRE(!data.has_key(3, 2));
    }

    {
        DistributedDataShared<std::string> data{};

        data.add_obj(100, 200, "test");
        REQUIRE(data.has_key(100, 200));
        REQUIRE(!data.has_key(200, 100));

        data.add_obj(100, 200, "another_test"); // Multiple objects for same key
        REQUIRE(data.has_key(100, 200));
    }
}

TestStart(
    Unittest,
    "shambase/DistributedDataShared::get_element_count",
    distributedDataSharedTests_get_element_count,
    1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};
        REQUIRE(data.get_element_count() == 0);

        data.add_obj(1, 2, 42);
        REQUIRE(data.get_element_count() == 1);

        data.add_obj(3, 4, 84);
        REQUIRE(data.get_element_count() == 2);

        data.add_obj(1, 2, 21); // Same patch pair, different object
        REQUIRE(data.get_element_count() == 3);
    }

    {
        DistributedDataShared<std::vector<int>> data{};
        data.add_obj(1, 2, std::vector<int>{1, 2, 3});
        data.add_obj(2, 3, std::vector<int>{4, 5, 6});
        REQUIRE(data.get_element_count() == 2);
    }
}

TestStart(Unittest, "shambase/DistributedDataShared::map", distributedDataSharedTests_map, 1) {
    using namespace shambase;

    {
        DistributedDataShared<int> int_data{};
        int_data.add_obj(1, 2, 42);
        int_data.add_obj(3, 4, 84);
        int_data.add_obj(1, 2, 21);

        auto string_data = int_data.map<std::string>([](u64 left, u64 right, int &value) {
            return std::to_string(value) + "_" + std::to_string(left) + "_" + std::to_string(right);
        });

        REQUIRE(string_data.get_element_count() == 3);
        REQUIRE(string_data.has_key(1, 2));
        REQUIRE(string_data.has_key(3, 4));

        // Verify original data is unchanged
        REQUIRE(int_data.get_element_count() == 3);
        REQUIRE(int_data.has_key(1, 2));
        REQUIRE(int_data.has_key(3, 4));

        // Check that mapped values are correct
        std::vector<std::string> collected_strings;
        string_data.for_each([&](u64 left, u64 right, std::string &value) {
            collected_strings.push_back(value);
        });

        REQUIRE(collected_strings.size() == 3);

        bool found_42_1_2 = false, found_84_3_4 = false, found_21_1_2 = false;
        for (const auto &str : collected_strings) {
            if (str == "42_1_2")
                found_42_1_2 = true;
            if (str == "84_3_4")
                found_84_3_4 = true;
            if (str == "21_1_2")
                found_21_1_2 = true;
        }
        REQUIRE(found_42_1_2);
        REQUIRE(found_84_3_4);
        REQUIRE(found_21_1_2);
    }

    {
        DistributedDataShared<double> double_data{};
        double_data.add_obj(5, 10, 3.14);

        auto int_data = double_data.map<int>([](u64 left, u64 right, double &value) {
            return static_cast<int>(value);
        });

        REQUIRE(int_data.get_element_count() == 1);
        REQUIRE(int_data.has_key(5, 10));

        int_data.for_each([](u64 left, u64 right, int &value) {
            REQUIRE(value == 3);
        });
    }
}

TestStart(
    Unittest,
    "shambase/DistributedDataShared::reset_and_is_empty",
    distributedDataSharedTests_reset_and_is_empty,
    1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};
        REQUIRE(data.is_empty());
        REQUIRE(data.get_element_count() == 0);

        data.add_obj(1, 2, 42);
        data.add_obj(3, 4, 84);
        REQUIRE(!data.is_empty());
        REQUIRE(data.get_element_count() == 2);

        data.reset();
        REQUIRE(data.is_empty());
        REQUIRE(data.get_element_count() == 0);
        REQUIRE(!data.has_key(1, 2));
        REQUIRE(!data.has_key(3, 4));
    }

    {
        DistributedDataShared<std::vector<int>> data{};
        data.add_obj(1, 2, std::vector<int>{1, 2, 3, 4, 5});
        REQUIRE(!data.is_empty());

        data.reset();
        REQUIRE(data.is_empty());
    }
}

TestStart(
    Unittest,
    "shambase/DistributedDataShared::get_native",
    distributedDataSharedTests_get_native,
    1) {
    using namespace shambase;

    {
        DistributedDataShared<int> data{};
        data.add_obj(1, 2, 42);
        data.add_obj(3, 4, 84);

        auto &native_map = data.get_native();
        REQUIRE(native_map.size() == 2);

        // Check direct access to multimap
        auto it = native_map.find({1, 2});
        REQUIRE(it != native_map.end());
        REQUIRE(it->second == 42);

        it = native_map.find({3, 4});
        REQUIRE(it != native_map.end());
        REQUIRE(it->second == 84);

        // Test that we can modify through native access
        it->second = 168;

        bool found_modified = false;
        data.for_each([&](u64 left, u64 right, int &value) {
            if (left == 3 && right == 4 && value == 168) {
                found_modified = true;
            }
        });
        REQUIRE(found_modified);
    }
}
