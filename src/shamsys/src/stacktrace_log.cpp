// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file stacktrace_log.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "shamsys/stacktrace_log.hpp"
#include <sstream>
#include <vector>

#ifdef SHAMROCK_USE_CPPTRACE
    #include <cpptrace/cpptrace.hpp>
    #include <cpptrace/formatting.hpp>
#endif

namespace {

    // replace rules
    static const std::vector<std::pair<std::string, std::string>> replace_rules = {
#ifdef SYCL_COMP_ACPP
        {"hipsycl::sycl::vec<long, 3, hipsycl::sycl::detail::vec_storage<long, 3> >", "i64_3"},
        {"hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >", "f64_3"},
        {"hipsycl::sycl::", "sycl::"},
#endif
#ifdef SYCL_COMP_DPCPP
        {"sycl::_V1::vec<long, 3>", "i64_3"},
        {"sycl::_V1::vec<double, 3>", "f64_3"},
        {"sycl::_V1::", "sycl::"},
#endif
    };

#ifdef SHAMROCK_USE_CPPTRACE
    static cpptrace::formatter formatter = {};
#endif

} // namespace

namespace shamsys {

    void init_backtrace_utilities(bool enable_colors) {

#ifdef SHAMROCK_USE_CPPTRACE
        auto color_mode = enable_colors ? cpptrace::formatter::color_mode::always
                                        : cpptrace::formatter::color_mode::none;

        formatter = cpptrace::formatter{}
                        .transform([](cpptrace::stacktrace_frame frame) {
                            for (const auto &[pattern, replacement] : replace_rules) {
                                shambase::replace_all(frame.symbol, pattern, replacement);
                            }
                            return frame;
                        })
                        .symbols(cpptrace::formatter::symbol_mode::pretty)
                        .colors(color_mode)
                        .break_before_filename()
                        .snippets(false);
#endif
    }

    std::string crash_report_backtrace() {
        std::stringstream ss;
        ss << "------ Profiler stacktrace ------\n";
        ss << "  - Provide a basic stacktrace based on the locations of StackEntry objects\n";
#ifndef SHAMROCK_USE_CPPTRACE
        ss << "  - To get the precise a more precise stacktrace, please reconfigure with :\n";
        ss << "    \"cmake . -DSHAMROCK_USE_CPPTRACE=on -DCMAKE_BUILD_TYPE=RelWithDebInfo\"\n";
#endif
        ss << "\n";
        ss << shambase::fmt_callstack();
        ss << "------ End of profiler stacktrace ------\n";
        ss << "\n";
#ifdef SHAMROCK_USE_CPPTRACE
        ss << "------ True stacktrace ------\n";
        ss << "  - Provide a true stacktrace based on the actual call stack (using cpptrace)\n";
        ss << "  - Please compile with -g to get line informations (cmake . "
              "-DCMAKE_BUILD_TYPE=RelWithDebInfo)\n";
        ss << "\n";
        ss << formatter.format(cpptrace::generate_trace());
        ss << "\n";
        ss << "------ End of true stacktrace ------\n";
#endif
        return ss.str();
    }

} // namespace shamsys
