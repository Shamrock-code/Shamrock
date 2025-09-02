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
 * @file assert.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shamrock assertion utility
 *
 */

#define ASSERT_MODE_CASSERT 1       ///< Use C assert for shamrock assertion
#define ASSERT_MODE_RUNTIME_ERROR 2 ///< Use C++ exception for shamrock assertion

#ifdef SHAM_ASSERT_IS
    #if SHAM_ASSERT_IS == ASSERT_MODE_RUNTIME_ERROR
        #include "shambase/exception.hpp"
    #endif
    #if SHAM_ASSERT_IS == ASSERT_MODE_CASSERT
        #include <cassert>
    #endif
#endif

/*
 * This file make heavy use of the legendary do while false trick
 * do { ... } while (false) in order to force a compilation error
 * for missing ; for the end of the macro call.
 */

/**
 * @brief Macro to assert that a condition is true
 */
#define SHAM_ASSERT_NAMED(message, condition)                                                      \
    do {                                                                                           \
    } while (false)

#ifdef SHAM_ASSERT_IS
    #undef SHAM_ASSERT_NAMED // we are about to redefine it

    #if SHAM_ASSERT_IS == ASSERT_MODE_CASSERT
        #define SHAM_ASSERT_NAMED(message, condition)                                              \
            do {                                                                                   \
                assert(((void) message, condition));                                               \
            } while (false)
    #elif SHAM_ASSERT_IS == ASSERT_MODE_RUNTIME_ERROR
        #define SHAM_ASSERT_NAMED(message, condition)                                              \
            do {                                                                                   \
                if (!(condition)) {                                                                \
                    shambase::throw_with_loc<std::runtime_error>(message);                         \
                }                                                                                  \
            } while (false)
    #else
        #error                                                                                     \
            "Unknown value for SHAM_ASSERT_IS. Possible values are: ASSERT_MODE_CASSERT,ASSERT_MODE_RUNTIME_ERROR"
    #endif

#endif

/// Shorthand for SHAM_ASSERT_NAMED without a message
#define SHAM_ASSERT(x) SHAM_ASSERT_NAMED(#x, x)
