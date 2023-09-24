// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "shambase/SourceLocation.hpp"

#include "TestAssert.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl.hpp"

namespace shamtest::details {
    struct TestAssertList {
        std::vector<TestAssert> asserts;

        // define member function here
        // to register asserts

        inline static std::string gen_comment(std::string s, SourceLocation loc) {
            return s + "\n" + loc.format_multiline();
        }

        inline void
        assert_bool(std::string assert_name, bool v, SourceLocation loc = SourceLocation{}) {

            asserts.push_back(TestAssert{v, std::move(assert_name), gen_comment("", loc)});
        }

        template<class T1, class T2>
        inline void
        assert_equal(std::string assert_name, T1 a, T2 b, SourceLocation loc = SourceLocation{}) {

            bool t              = a == b;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b);
            }

            asserts.push_back(TestAssert{t, std::move(assert_name), gen_comment(comment, loc)});
        }


        template<class Acca,class Accb>
        inline void
        assert_equal_array(std::string assert_name, Acca & acc_a, Accb & acc_b, u32 len, SourceLocation loc = SourceLocation{}) {

            bool t = true;std::string comment = "";

            for(u32 i = 0; i < len; i++){
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

        inline void assert_float_equal(
            std::string assert_name, f64 a, f64 b, f64 eps, SourceLocation loc = SourceLocation{}
        ) {
            f64 diff = sycl::fabs(a - b);

            bool t              = diff < eps;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b) +
                          " diff=" + std::to_string(diff);
            }

            asserts.push_back(TestAssert{t, std::move(assert_name), gen_comment(comment, loc)});
        }

        inline void assert_add_comment(
            std::string assert_name,
            bool v,
            std::string comment,
            SourceLocation loc = SourceLocation{}
        ) {
            asserts.push_back(TestAssert{v, std::move(assert_name), gen_comment(comment, loc)});
        }

        std::string serialize();
    };
} // namespace shamtest::details