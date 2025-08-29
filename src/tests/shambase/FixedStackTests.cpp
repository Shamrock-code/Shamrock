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

TestStart(Unittest, "shambase/FixedStack", fixed_stack_default_constructor, 1) {

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
        REQUIRE_EQUAL(stack.pop(), initial_value);
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
        u32 popped1 = stack.pop();
        REQUIRE_EQUAL(popped1, 30);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        u32 popped2 = stack.pop();
        REQUIRE_EQUAL(popped2, 20);
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        u32 popped3 = stack.pop();
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
        REQUIRE_EQUAL(stack.pop(), 300);
        REQUIRE_EQUAL(stack.pop(), 200);
        REQUIRE_EQUAL(stack.pop(), 100);

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
        u32 val = stack.pop();
        REQUIRE_EQUAL(val, 2);

        // Push more
        stack.push(3);
        stack.push(4);

        // Check state
        REQUIRE_EQUAL(stack.is_not_empty(), true);

        // Pop remaining values
        REQUIRE_EQUAL(stack.pop(), 4);
        REQUIRE_EQUAL(stack.pop(), 3);
        REQUIRE_EQUAL(stack.pop(), 1);

        REQUIRE_EQUAL(stack.is_not_empty(), false);
    }
}
