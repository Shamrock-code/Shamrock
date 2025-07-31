// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file WithUUID.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/WithUUID.hpp"
#include "shamtest/shamtest.hpp"
#include <unordered_set>
#include <thread>

template<bool thread_safe>
void test() {

    // check that it reset to 0
    class A1 : public shambase::WithUUID<A1, u32, thread_safe> {};
    class A2 : public shambase::WithUUID<A2, u32, thread_safe> {};

    A1 a1;
    A2 a2;

    REQUIRE(a1.get_uuid() == a2.get_uuid());

    // test that there is no duplicate
    const int numInstances = 100;
    std::unordered_set<u64> uuidSet;
    for (int i = 0; i < numInstances; ++i) {
        A1 instance;
        auto uuid = instance.get_uuid();
        REQUIRE(uuidSet.find(uuid) == uuidSet.end());
        uuidSet.insert(uuid);
    }

    // multithreaded case
    if constexpr (thread_safe) {
        class A3 : public shambase::WithUUID<A3, u32, thread_safe> {};

        // test that there is no duplicate when creating in parallel
        const int numThreads = 10; // should be a divisor of numInstances
        std::vector<std::thread> threads(numThreads);
        std::atomic<int> counter(0);
        std::unordered_set<u64> uuidSet;
        std::mutex uuidSetMutex;
        for (int i = 0; i < numThreads; ++i) {
            threads[i] = std::thread([&uuidSet, &uuidSetMutex, &counter]() {
                for (int j = 0; j < numInstances / numThreads; ++j) {
                    A3 instance;
                    auto uuid = instance.get_uuid();
                    std::lock_guard<std::mutex> lock(uuidSetMutex);
                    auto it = uuidSet.find(uuid);
                    REQUIRE(it == uuidSet.end());
                    uuidSet.insert(uuid);
                    ++counter;
                }
            });
        }
        for (auto &t : threads) {
            t.join();
        }
        REQUIRE(counter == numInstances);
    }
}

TestStart(Unittest, "shambase/WithUUID(t-unsafe)", with_uuid_tunsafe, 1) { test<false>(); }

TestStart(Unittest, "shambase/WithUUID(safe)", with_uuid_tsafe, 1) { test<true>(); }
