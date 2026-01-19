// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file FixedStack.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Fixed-size stack container for high-performance applications
 *
 * This header provides a stack container with compile-time fixed capacity,
 * designed for performance-critical applications where dynamic memory allocation
 * must be avoided. The stack uses a statically allocated array and manages
 * elements using a cursor-based approach.
 *
 * The FixedStack is particularly useful in GPU kernels, real-time systems,
 * and algorithms where memory allocation overhead is prohibitive, such as
 * tree traversal algorithms or depth-first search operations.
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include <type_traits>

namespace shambase {

    /**
     * @brief Fixed-capacity stack container with compile-time size determination
     *
     * A high-performance stack implementation that uses a statically allocated
     * array for storage. The stack capacity is determined at compile time,
     * eliminating runtime memory allocation overhead. The implementation uses
     * a cursor-based approach where the cursor points to the next available
     * slot in the stack array.
     *
     * The stack grows downward in memory (decreasing indices), with the cursor
     * starting at stack_size (empty stack) and decreasing as elements are pushed.
     * This design choice optimizes for common usage patterns in computational
     * algorithms.
     *
     * @tparam T Element type to store in the stack
     * @tparam stack_size Maximum number of elements the stack can hold
     *
     * @note The stack array is intentionally not zero-initialized in the default
     *       constructor to avoid unnecessary initialization overhead in performance-
     *       critical applications.
     *
     * @code{.cpp}
     * // Example: Basic stack operations
     * shambase::FixedStack<u32, 10> stack;
     *
     * // Push elements
     * stack.push(42);
     * stack.push(17);
     * stack.push(8);
     *
     * // Check if stack has elements
     * while (stack.is_not_empty()) {
     *     u32 value = stack.pop_ret();
     *     // Process value (8, 17, 42 in LIFO order)
     * }
     *
     * // Example: Initialize with a value
     * shambase::FixedStack<u32, 5> initialized_stack(100);
     * // Stack contains one element: 100
     *
     * // Example: Use in tree traversal
     * shambase::FixedStack<u32, 64> node_stack;
     * node_stack.push(root_node_id);
     *
     * while (node_stack.is_not_empty()) {
     *     u32 current_node = node_stack.pop_ret();
     *     // Process current node
     *     // Push child nodes for further processing
     *     for (u32 child : get_children(current_node)) {
     *         node_stack.push(child);
     *     }
     * }
     * @endcode
     */
    template<class T, u32 stack_size>
    struct FixedStack {

        static_assert(
            std::is_trivially_destructible_v<T>,
            "FixedStack only supports trivially destructible types to avoid resource leaks.");
        static_assert(stack_size > 0, "FixedStack must have a size greater than 0.");

        /// Storage array for stack elements
        T id_stack[stack_size];
        /// Cursor pointing to the next available slot (stack_size = empty, 0 = full)
        u32 stack_cursor;

        // Note that the stack itself is voluntarily not initialized
        // do not add it to the constructor otherwise we may have to pay for zero initialization

        /// Default constructor creating an empty stack
        inline constexpr FixedStack() : stack_cursor{stack_size} {}

        /// Constructor creating a stack with one initial element
        inline constexpr FixedStack(const T &val) : stack_cursor{stack_size - 1} {
            id_stack[stack_cursor] = val;
        }

        /// Check if the stack contains any elements
        inline constexpr bool is_not_empty() const { return stack_cursor < stack_size; }

        /// Check if the stack is empty
        inline constexpr bool empty() const { return stack_cursor == stack_size; }

        /// Returns the number of elements in the stack
        inline constexpr u32 size() const { return stack_size - stack_cursor; }

        /// Push an element onto the top of the stack
        inline void push(const T &val) {

            // FixedStack overflow: cannot push to a full stack.
            SHAM_ASSERT(stack_cursor > 0);

            stack_cursor--;
            id_stack[stack_cursor] = val;
        }

        /// Access the top element
        inline T &top() {
            SHAM_ASSERT(is_not_empty());
            return id_stack[stack_cursor];
        }

        /// Access the top element (const version)
        inline constexpr const T &top() const {
            SHAM_ASSERT(is_not_empty());
            return id_stack[stack_cursor];
        }

        /// Remove the top element from the stack
        inline constexpr void pop() {

            // FixedStack underflow: cannot pop from an empty stack.
            SHAM_ASSERT(is_not_empty());

            stack_cursor++;
        }

        /// Remove and return the top element from the stack
        [[nodiscard]]
        inline T pop_ret() {
            T val = top();
            pop();
            return val;
        }
    };

} // namespace shambase
