// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file FixedStack.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 *
 * @brief
 */

#include "shambase/aliases_int.hpp"

namespace shambase {

    template<class T, u32 stack_size>
    struct FixedStack {
        T id_stack[stack_size];
        u32 stack_cursor;

        // Note that the stack it self is volontarely not initialized
        // do not add it to the constructor otherwise we may have to pay for zero initialization

        FixedStack() : stack_cursor{stack_size} {}
        FixedStack(T val) : stack_cursor{stack_size - 1} { id_stack[stack_cursor] = val; }

        inline bool is_not_empty() const { return stack_cursor < stack_size; }

        inline void push(T val) {
            stack_cursor--;
            id_stack[stack_cursor] = val;
        }

        inline T pop() {
            T val = id_stack[stack_cursor];
            stack_cursor++;
            return val;
        }
    };

} // namespace shambase
