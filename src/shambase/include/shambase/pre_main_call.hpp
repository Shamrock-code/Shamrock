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
 * @file pre_main_call.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Provides a macro to call a lambda before main
 *
 */

#include "shambase/unique_name_macro.hpp"

/// Macro to call a lambda or a function before main
#define PRE_MAIN_FUNCTION_CALL_NAMED(class_name, instance_name, lambda)                            \
    namespace {                                                                                    \
        struct class_name {                                                                        \
            class_name() { lambda(); }                                                             \
        };                                                                                         \
        static const class_name instance_name{};                                                   \
    }

/**
 * @brief Macro to call a lambda or a function before main with a automatically generated unique
 * name
 *
 * Usage :
 * @code{.cpp}
 * int pre_main_call_counter = 0;
 * PRE_MAIN_FUNCTION_CALL([&]() {
 *     pre_main_call_counter++;
 * });
 * PRE_MAIN_FUNCTION_CALL(pre_main_call_function);
 * @endcode
 *
 * @param lambda The lambda or function to call
 */
#define PRE_MAIN_FUNCTION_CALL(lambda)                                                             \
    PRE_MAIN_FUNCTION_CALL_NAMED(                                                                  \
        __shamrock_unique_name(_PRE_MAIN_FUNCTION_CALL),                                           \
        __shamrock_unique_name(_PRE_MAIN_FUNCTION_CALL_INSTANCE),                                  \
        lambda)
