// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file shamtest.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief main include file for testing
 */

#include "details/Test.hpp"

/**
 * @brief namespace containing stuff related to the test library
 *
 */
namespace shamtest {

    /**
     * @brief implementation details of the test library
     *
     */
    namespace details {

        /**
         * @brief Static init vector containing the list of all the tests in the code
         * see : programming guide : Static init function registering
         */
        inline std::vector<Test> static_init_vec_tests{};

        /**
         * @brief helper class to statically register tests
         *
         */
        struct TestStaticInit {

            /**
             * @brief This constructor register the given arguments into `static_init_vec_tests`
             *
             * @param t the test info
             */
            inline explicit TestStaticInit(Test t) {
                static_init_vec_tests.push_back(std::move(t));
            }
        };

        /**
         * @brief the test currently running
         *
         */
        extern TestResult current_test;

    } // namespace details

    /// Configuration of the test runner
    struct TestConfig {

        /// Should print test list and then exit
        bool print_test_list_exit = false;

        /// Should display all logs including all asserts
        bool full_output = false;

        /// Should output a tex report
        bool output_tex = true;

        /// Should output a json report
        std::optional<std::string> json_output = {};

        bool run_long_tests = false; ///< run also long tests
        bool run_unittest   = true;  ///< run unittests
        bool run_validation = true;  ///< run validation tests
        bool run_benchmark  = false; ///< run benchmarks

        std::optional<std::string> run_only = {}; ///< Run only regex to select tests
    };

    /**
     * @brief run all the tests
     *
     * @param argc main argc
     * @param argv  main argv
     * @param cfg test run configuration
     * @return int exit code
     */
    int run_all_tests(int argc, char *argv[], TestConfig cfg);

    /// output test list to a file
    void gen_test_list(std::string_view outfile);

    /**
     * @brief current test asserts
     *
     * @return shamtest::details::TestAssertList& reference to the test asserts
     */
    inline shamtest::details::TestAssertList &asserts() {
        return shamtest::details::current_test.asserts;
    };

    /**
     * @brief current test data
     *
     * @return shamtest::details::TestAssertList& reference to the test datas
     */
    inline shamtest::details::TestDataList &test_data() {
        return shamtest::details::current_test.test_data;
    };

    /// Get current tex output from a test
    inline std::string &test_tex_out() { return shamtest::details::current_test.tex_output; }

} // namespace shamtest

/**
 * @brief Macro to declare a test
 *
 * Exemple :
 * \code{.cpp}
 * TestStart(Unittest, "testname", testfuncname, 1) {
 *     shamtest::asserts().assert_bool("what a reliable test", true);
 * }
 * \endcode
 */
#define TestStart(type, name, func_name, node_cnt)                                                 \
    void test_func_##func_name();                                                                  \
    void (*test_func_ptr_##func_name)() = test_func_##func_name;                                   \
    shamtest::details::TestStaticInit test_class_obj_##func_name(                                  \
        shamtest::details::Test{type, name, node_cnt, test_func_ptr_##func_name});                 \
    void test_func_##func_name()

/**
 * @brief Macro to write stuff to the tex test report
 *
 * Usage :
 * \code{.cpp}
 * TEX_REPORT(R"==(
 *   here i'm writing tex
 * )==")
 * \endcode
 */
#define TEX_REPORT(src) shamtest::details::current_test.tex_output += src;

#define STDSTRINGIFY(x) std::string(#x)

///////////////////////////////////////////////////////////////////////////////////////////////////
// Assert macros
///////////////////////////////////////////////////////////////////////////////////////////////////

// Note : the do-while are here to enforce the presence of a semicolumn after the call to the macros

namespace shamtest::details {
    /**
     *@brief Format a string that is an assert name
     *
     * If the string is empty, returns an empty string
     * Otherwise, returns the string with " | " appended
     */
    inline std::string format_assert_name(std::string s) {
        if (s == "") {
            return "";
        }
        return "\"" + s + "\" : ";
    }
} // namespace shamtest::details

/**
 * @brief Assert macro for test
 * write the conditional, the name of the assert will be the condition
 * Named variant
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_NAMED("assert name",a == 0)
 * \endcode
 */
#define REQUIRE_NAMED(name, a)                                                                     \
    do {                                                                                           \
        using namespace shamtest::details;                                                         \
        bool eval               = a;                                                               \
        std::string assert_name = format_assert_name(name) + #a;                                   \
        if (eval) {                                                                                \
            shamtest::asserts().assert_bool_with_log(assert_name, eval, "");                       \
        } else {                                                                                   \
            shamtest::asserts().assert_bool_with_log(                                              \
                assert_name,                                                                       \
                eval,                                                                              \
                STDSTRINGIFY(a) + " evaluated to false\n\n"                                        \
                    + " -> location : " + SourceLocation{}.format_one_line());                     \
        }                                                                                          \
    } while (0)

/**
 * @brief Assert macro for test to test for equalities using a custom comparison function
 * Named variant
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_EQUAL_CUSTOM_COMP_NAMED("assert name",a , b)
 * \endcode
 */
#define REQUIRE_EQUAL_CUSTOM_COMP_NAMED(name, _a, _b, comp)                                        \
    do {                                                                                           \
        auto a = _a;                                                                               \
        auto b = _b;                                                                               \
        using namespace shamtest::details;                                                         \
        bool eval               = comp(a, b);                                                      \
        std::string assert_name = format_assert_name(name) + #_a " == " #_b;                       \
        if (eval) {                                                                                \
            shamtest::asserts().assert_bool_with_log(assert_name, eval, "");                       \
        } else {                                                                                   \
            shamtest::asserts().assert_bool_with_log(                                              \
                assert_name,                                                                       \
                eval,                                                                              \
                assert_name + " evaluated to false\n\n" + shambase::format(" -> " #_a " = {}", a)  \
                    + "\n" + shambase::format(" -> " #_b " = {}", b) + "\n"                        \
                    + " -> location : " + SourceLocation{}.format_one_line());                     \
        }                                                                                          \
    } while (0)

/**
 * @brief Assert macro for test to test for equalities
 * Named variant
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_EQUAL_NAMED("assert_name", a , b)
 * \endcode
 */
#define REQUIRE_EQUAL_NAMED(name, a, b)                                                            \
    REQUIRE_EQUAL_CUSTOM_COMP_NAMED(name, a, b, [](const auto &p1, const auto &p2) {               \
        return p1 == p2;                                                                           \
    })

/**
 * @brief Assert macro for test, testing equality between two variables, with a given precision and
 * a custom distance function
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("assert name",a , b, 1e-9, sycl::length)
 * \endcode
 */
#define REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, _a, _b, prec, dist)                            \
    do {                                                                                           \
        auto a = _a;                                                                               \
        auto b = _b;                                                                               \
        using namespace shamtest::details;                                                         \
        bool eval = dist((a) - (b)) < prec;                                                        \
        std::string assert_name                                                                    \
            = format_assert_name(name) + #dist "(" #_a ") - (" #_b ") < " #prec;                   \
        if (eval) {                                                                                \
            shamtest::asserts().assert_bool_with_log(assert_name, eval, "");                       \
        } else {                                                                                   \
            shamtest::asserts().assert_bool_with_log(                                              \
                assert_name,                                                                       \
                eval,                                                                              \
                assert_name + " evaluated to false\n\n" + shambase::format(" -> " #_a " = {}", a)  \
                    + "\n" + shambase::format(" -> " #_b " = {}", b) + "\n"                        \
                    + shambase::format(" -> " #prec " = {}", prec) + "\n"                          \
                    + " -> location : " + SourceLocation{}.format_one_line());                     \
        }                                                                                          \
    } while (0)

/**
 * @brief Assert macro for test, testing equality between two variables, with a given precision
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_FLOAT_EQUAL(a , b, 1e-9)
 * \endcode
 */
#define REQUIRE_FLOAT_EQUAL_NAMED(name, a, b, prec)                                                \
    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED(name, a, b, prec, std::abs)

/**
 * @brief Assert macro for test
 * write the conditional, the name of the assert will be the condition
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE(a == 0)
 * \endcode
 */
#define REQUIRE(a) REQUIRE_NAMED("", a)

/**
 * @brief Assert macro for test to test for equalities using a custom comparison function
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_EQUAL_CUSTOM_COMP(a , b)
 * \endcode
 */
#define REQUIRE_EQUAL_CUSTOM_COMP(a, b, comp) REQUIRE_EQUAL_CUSTOM_COMP_NAMED("", a, b, comp)

/**
 * @brief Assert macro for test to test for equalities
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_EQUAL(a , b)
 * \endcode
 */
#define REQUIRE_EQUAL(a, b) REQUIRE_EQUAL_NAMED("", a, b)

/**
 * @brief Assert macro for test, testing equality between two variables, with a given precision and
 * a custom distance function
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_FLOAT_EQUAL_CUSTOM_DIST(a , b, 1e-9, sycl::length)
 * \endcode
 */
#define REQUIRE_FLOAT_EQUAL_CUSTOM_DIST(name, a, b, prec, dist)                                    \
    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", a, b, prec, dist)

/**
 * @brief Assert macro for test, testing equality between two variables, with a given precision
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_FLOAT_EQUAL(a , b, 1e-9)
 * \endcode
 */
#define REQUIRE_FLOAT_EQUAL(a, b, prec)                                                            \
    REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", a, b, prec, std::abs)

/**
 * @brief Assert macro for test, testing that a given call throws a specific exception type
 *
 * Usage :
 * \code{.cpp}
 * REQUIRE_EXCEPTION_THROW(function_that_throws(), exception_type)
 * \endcode
 *
 * @param call Call that is expected to throw the specified exception type
 * @param exception_type Exception type that is expected to be thrown
 */
#define REQUIRE_EXCEPTION_THROW(call, exception_type)                                              \
    do {                                                                                           \
        try {                                                                                      \
            /* Try to call the function that is expected to throw */                               \
            call;                                                                                  \
            /* If no exception is thrown, assert that the test failed */                           \
            shamtest::asserts().assert_bool_with_log(                                              \
                #exception_type " was not thrown",                                                 \
                false,                                                                             \
                "Expected throw of type " #exception_type ", but nothing was thrown\n"             \
                " -> location : "                                                                  \
                    + SourceLocation{}.format_one_line());                                         \
        } catch (const exception_type &ex) {                                                       \
            /* If wanted exception is thrown, assert that the test pass */                         \
            shamtest::asserts().assert_bool_with_log(                                              \
                "Found wanted throw of type " #exception_type, true, "");                          \
        } catch (const std::exception &e) {                                                        \
            /* If another exception type is thrown, assert that the test failed */                 \
            shamtest::asserts().assert_bool_with_log(                                              \
                #exception_type " was not thrown",                                                 \
                false,                                                                             \
                "Expected throw of type " #exception_type ", but got " + std::string(e.what())     \
                    + "\n" + " -> location : " + SourceLocation{}.format_one_line());              \
        } catch (...) {                                                                            \
            /* If an unknown exception is thrown, assert that the test failed */                   \
            shamtest::asserts().assert_bool_with_log(                                              \
                #exception_type " was not thrown",                                                 \
                false,                                                                             \
                "Expected throw of type " #exception_type ", but got unknown exception\n"          \
                " -> location : "                                                                  \
                    + SourceLocation{}.format_one_line());                                         \
        }                                                                                          \
    } while (0)
