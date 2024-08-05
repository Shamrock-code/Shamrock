// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Test.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
