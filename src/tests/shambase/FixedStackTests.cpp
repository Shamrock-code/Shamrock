// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/FixedStack.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/FixedStack", shambase_fixedstack_test, 1) {

    // Test default constructor
    {
        constexpr u32 stack_size = 5;
        shambase::FixedStack<u32, stack_size> stack;

        // Stack should be empty (cursor at stack_size means empty)
        REQUIRE_EQUAL(stack.is_not_empty(), false);
        REQUIRE_EQUAL(stack.stack_cursor, stack_size);
    }

    // Test constructor with initial value
    {
        constexpr u32 stack_size = 5;
        u32 initial_value        = 42;
        shambase::FixedStack<u32, stack_size> stack(initial_value);

        // Stack should not be empty and contain the initial value
        REQUIRE_EQUAL(stack.is_not_empty(), true);
        REQUIRE_EQUAL(stack.pop_ret(), initial_value);
        REQUIRE_EQUAL(stack.is_not_empty(), false);
    }

    // Test basic push and pop operations
    {
        constexpr u32 stack_size = 4;
        shambase::FixedStack<u32, stack_size> stack;

        REQUIRE_EQUAL(stack.is_not_empty(), false);

        // Push values
        stack.push(10);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        stack.push(20);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        stack.push(30);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        // Pop values
        u32 popped1 = stack.pop_ret();
        REQUIRE_EQUAL(popped1, 30);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        u32 popped2 = stack.pop_ret();
        REQUIRE_EQUAL(popped2, 20);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        u32 popped3 = stack.pop_ret();
        REQUIRE_EQUAL(popped3, 10);
        REQUIRE_EQUAL(stack.is_not_empty(), false);
    }

    // Test filling the stack completely
    {
        constexpr u32 stack_size = 3;
        shambase::FixedStack<u32, stack_size> stack;

        // Fill stack completely
        stack.push(100);
        stack.push(200);
        stack.push(300);

        REQUIRE_EQUAL(stack.is_not_empty(), true);
        REQUIRE_EQUAL(stack.stack_cursor, 0u);

        // Pop all values
        REQUIRE_EQUAL(stack.pop_ret(), 300);
        REQUIRE_EQUAL(stack.pop_ret(), 200);
        REQUIRE_EQUAL(stack.pop_ret(), 100);

        REQUIRE_EQUAL(stack.is_not_empty(), false);
    }

    // Test mixed push/pop operations
    {
        constexpr u32 stack_size = 5;
        shambase::FixedStack<u32, stack_size> stack;

        // Push some values
        stack.push(1);
        stack.push(2);

        // Pop one
        u32 val = stack.pop_ret();
        REQUIRE_EQUAL(val, 2);

        // Push more
        stack.push(3);
        stack.push(4);

        // Check state
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        // Pop remaining values
        REQUIRE_EQUAL(stack.pop_ret(), 4);
        REQUIRE_EQUAL(stack.pop_ret(), 3);
        REQUIRE_EQUAL(stack.pop_ret(), 1);

        REQUIRE_EQUAL(stack.is_not_empty(), false);
    }

    // Test empty() function
    {
        constexpr u32 stack_size = 4;
        shambase::FixedStack<u32, stack_size> stack;

        // Stack should be empty initially
        REQUIRE_EQUAL(stack.empty(), true);
        REQUIRE_EQUAL(stack.is_not_empty(), false);

        // After pushing one element, should not be empty
        stack.push(42);
        REQUIRE_EQUAL(stack.empty(), false);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        // After pushing more elements, should still not be empty
        stack.push(100);
        stack.push(200);
        REQUIRE_EQUAL(stack.empty(), false);

        // Pop all elements except one
        stack.pop();
        stack.pop();
        REQUIRE_EQUAL(stack.empty(), false);

        // Pop last element, should be empty again
        stack.pop();
        REQUIRE_EQUAL(stack.empty(), true);
    }

    // Test size() function
    {
        constexpr u32 stack_size = 5;
        shambase::FixedStack<u32, stack_size> stack;

        // Empty stack should have size 0
        REQUIRE_EQUAL(stack.size(), 0u);

        // Push elements and check size increases
        stack.push(10);
        REQUIRE_EQUAL(stack.size(), 1u);

        stack.push(20);
        REQUIRE_EQUAL(stack.size(), 2u);

        stack.push(30);
        REQUIRE_EQUAL(stack.size(), 3u);

        stack.push(40);
        REQUIRE_EQUAL(stack.size(), 4u);

        stack.push(50);
        REQUIRE_EQUAL(stack.size(), 5u); // Full stack

        // Pop elements and check size decreases
        stack.pop();
        REQUIRE_EQUAL(stack.size(), 4u);

        stack.pop();
        REQUIRE_EQUAL(stack.size(), 3u);

        stack.pop();
        stack.pop();
        stack.pop();
        REQUIRE_EQUAL(stack.size(), 0u); // Empty again
    }

    // Test top() function
    {
        constexpr u32 stack_size = 4;
        shambase::FixedStack<u32, stack_size> stack;

        // Push elements and test top access
        stack.push(100);
        REQUIRE_EQUAL(stack.top(), 100u);

        stack.push(200);
        REQUIRE_EQUAL(stack.top(), 200u);

        stack.push(300);
        REQUIRE_EQUAL(stack.top(), 300u);

        // Modify top element through reference
        stack.top() = 999;
        REQUIRE_EQUAL(stack.top(), 999u);

        // Pop and check top changes
        stack.pop();
        REQUIRE_EQUAL(stack.top(), 200u);

        stack.pop();
        REQUIRE_EQUAL(stack.top(), 100u);
    }

    // Test top() const version
    {
        constexpr u32 stack_size = 3;
        shambase::FixedStack<u32, stack_size> stack;

        stack.push(42);
        stack.push(84);

        const auto &const_stack = stack;
        REQUIRE_EQUAL(const_stack.top(), 84u);

        stack.pop();
        REQUIRE_EQUAL(const_stack.top(), 42u);
    }

    // Test constructor with initial value using new functions
    {
        constexpr u32 stack_size = 3;
        u32 initial_value        = 777;
        shambase::FixedStack<u32, stack_size> stack(initial_value);

        REQUIRE_EQUAL(stack.empty(), false);
        REQUIRE_EQUAL(stack.size(), 1u);
        REQUIRE_EQUAL(stack.top(), initial_value);

        // Modify through top reference and verify
        stack.top() = 888;
        REQUIRE_EQUAL(stack.top(), 888u);
        REQUIRE_EQUAL(stack.size(), 1u);
    }
}
