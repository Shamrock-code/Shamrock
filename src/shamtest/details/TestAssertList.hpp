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
 * @file TestAssertList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/string.hpp"
#include "TestAssert.hpp"
#include "shambackends/sycl.hpp"

namespace shamtest::details {

    /// Class to hold the list of assertion related to a test
    struct TestAssertList {

        /// List of assertion held by the class
        std::vector<TestAssert> asserts;

        // define member function here
        // to register asserts

        inline void assert_bool_with_log(std::string assert_name, bool v, std::string log) {
            asserts.push_back(TestAssert{v, std::move(assert_name), std::move(log)});
        }

        /// Append the source location to the the supplied string to generate a comment
        inline static std::string gen_comment(std::string s, SourceLocation loc) {
            return s + "\n" + loc.format_multiline();
        }

        /// Test if the supplied boolean is true
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_bool(
            std::string assert_name, bool v, SourceLocation loc = SourceLocation{}) {

            asserts.push_back(
                TestAssert{
                    v,
                    std::move(assert_name),
                    "failed assert location : " + loc.format_one_line()});
        }

        /// Test for an equality
        template<class T1, class T2>
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_equal(
            std::string assert_name, T1 a, T2 b, SourceLocation loc = SourceLocation{}) {

            bool t              = a == b;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b);
            }

            asserts.push_back(TestAssert{t, std::move(assert_name), gen_comment(comment, loc)});
        }

        /// Assert equal on an array of values
        template<class Acca, class Accb>
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_equal_array(
            std::string assert_name,
            Acca &acc_a,
            Accb &acc_b,
            u32 len,
            SourceLocation loc = SourceLocation{}) {

            bool t              = true;
            std::string comment = "";

            for (u32 i = 0; i < len; i++) {
                t = t && (acc_a[i] == acc_b[i]);
            }

            if (!t) {
                comment += "left : \n";
                comment += shambase::format_array(acc_a, len, 16, "{} ");
                comment += "right : \n";
                comment += shambase::format_array(acc_b, len, 16, "{} ");
            }

            asserts.push_back(TestAssert{t, std::move(assert_name), gen_comment(comment, loc)});
        }

        /**
         * @brief Add an assertion testing a floating point equality up to precision eps
         *
         * @param assert_name name of the assertion
         * @param a value a
         * @param b value b
         * @param eps precision of the test
         * @param loc source location of the call
         */
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_float_equal(
            std::string assert_name, f64 a, f64 b, f64 eps, SourceLocation loc = SourceLocation{}) {
            f64 diff = sycl::fabs(a - b);

            bool t              = diff < eps;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b)
                          + " diff=" + std::to_string(diff);
            }

            asserts.push_back(TestAssert{t, std::move(assert_name), gen_comment(comment, loc)});
        }

        /// add an assertion with a comment
        inline void assert_add_comment(
            std::string assert_name,
            bool v,
            std::string comment,
            SourceLocation loc = SourceLocation{}) {
            asserts.push_back(TestAssert{v, std::move(assert_name), gen_comment(comment, loc)});
        }

        /// Serialize the assertion in JSON
        std::string serialize_json();
        /// Serialize the assertion in binary format
        void serialize(std::basic_stringstream<byte> &stream);
        /// DeSerialize the assertion from binary format
        static TestAssertList deserialize(std::basic_stringstream<byte> &reader);

        /// Get number of assertion in the list
        inline u32 get_assert_count() { return asserts.size(); }

        /// Get the number of successfull assertions
        inline u32 get_assert_success_count() {
            u32 cnt = 0;
            for (TestAssert &a : asserts) {
                if (a.value) {
                    cnt++;
                }
            }
            return cnt;
        }
    };
} // namespace shamtest::details
