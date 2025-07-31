// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/stacktrace.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/stacktrace/print_stack", test_stackprinter, 1) {
    StackEntry stack{};

    logger::raw_ln(shambase::fmt_callstack());

    // throw shambase::make_except_with_loc<std::invalid_argument>("test");
}
