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
 * @file Test.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "TestResult.hpp"

namespace shamtest::details {

    /// Current test being run
    extern TestResult current_test;

    /// Informations about a test
    struct Test {
        TestType type;          ///< Type of test
        std::string name;       ///< Name of the test
        i32 node_count;         ///< Node count of the test
        void (*test_functor)(); ///< Test function

        /// CTOR of the test
        inline Test(const TestType &type, std::string name, const i32 &node_count, void (*func)())
            : type(type), name(std::move(name)), node_count(node_count), test_functor(func) {}

        /// Run the test
        TestResult run();
    };

} // namespace shamtest::details
