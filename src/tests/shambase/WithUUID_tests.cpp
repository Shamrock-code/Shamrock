// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include <type_traits>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <utility>

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

template<bool thread_safe>
void test_move_invalidate() {

    class B1 : public shambase::WithUUID<B1, u32, thread_safe> {};

    // Copying would duplicate the uuid across two live instances, so it must be disallowed
    // entirely rather than merely discouraged.
    static_assert(!std::is_copy_constructible_v<B1>, "WithUUID must not be copy-constructible");
    static_assert(!std::is_copy_assignable_v<B1>, "WithUUID must not be copy-assignable");

    B1 a;
    u32 a_uuid = a.get_uuid();
    REQUIRE(a.is_alive());
    REQUIRE(a_uuid != decltype(a)::invalid_uuid);

    // Move construction transfers the uuid to the destination and invalidates the source.
    B1 b(std::move(a));
    REQUIRE(b.get_uuid() == a_uuid);
    REQUIRE(b.is_alive());
    // NOLINTBEGIN(bugprone-use-after-move): checking the moved-from state is the point here.
    REQUIRE(a.is_alive() == false);
    REQUIRE(a.get_uuid() == decltype(a)::invalid_uuid);
    // NOLINTEND(bugprone-use-after-move)

    // Move assignment does the same: transfers the uuid over, invalidates the source, on top of
    // whatever uuid the destination held before (which is simply discarded here -- WithUUID
    // itself has no notion of "this uuid is going away", that is up to classes built on top of
    // it, e.g. LifetimeTracker).
    B1 c;
    REQUIRE(c.is_alive());
    c = std::move(b);
    REQUIRE(c.get_uuid() == a_uuid);
    REQUIRE(c.is_alive());
    // NOLINTBEGIN(bugprone-use-after-move): checking the moved-from state is the point here.
    REQUIRE(b.is_alive() == false);
    REQUIRE(b.get_uuid() == decltype(b)::invalid_uuid);
    // NOLINTEND(bugprone-use-after-move)

    // Self move-assignment must not invalidate the only instance holding the uuid.
    c = std::move(c);
    REQUIRE(c.is_alive());
    REQUIRE(c.get_uuid() == a_uuid);
}

NEW_TEST(Unittest, "shambase/WithUUID(t-unsafe)", 1) {
    test<false>();
    test_move_invalidate<false>();
}

NEW_TEST(Unittest, "shambase/WithUUID(safe)", 1) {
    test<true>();
    test_move_invalidate<true>();
}
