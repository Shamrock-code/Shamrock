// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/pre_main_call.hpp"
#include "shamtest/shamtest.hpp"

namespace {
    int pre_main_call_counter = 0;
}

void pre_main_call_function() { pre_main_call_counter++; }

PRE_MAIN_FUNCTION_CALL(pre_main_call_function);

PRE_MAIN_FUNCTION_CALL([&]() {
    pre_main_call_counter++;
});

TestStart(Unittest, "shambase/pre_main_call", testpremaincall, 1) {
    REQUIRE_EQUAL(pre_main_call_counter, 2);
}
