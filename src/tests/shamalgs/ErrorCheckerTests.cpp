// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/atomic/ErrorChecker.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/atomic/ErrorCheckerFlags", test_ErrorCheckerFlags, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    enum ErrorCodes : u32 {
        Flag1 = 1 << 0,
        Flag2 = 1 << 1,
        Flag3 = 1 << 2,
    };

    shamalgs::atomic::ErrorCheckerFlags error_util(sched);

    sham::kernel_call(
        sched->get_queue(),
        sham::MultiRef{},
        sham::MultiRef{error_util},
        100,
        [](u32 i, auto error_util) {
            if (i == 2) {
                error_util.set_flag_on(Flag1);
            }
            if (i == 23 || i == 47) {
                error_util.set_flag_on(Flag2);
            }
        });

    u32 precondition_error = error_util.get_output();

    bool Flag_1_on = shambase::is_flag_on<Flag1>(precondition_error);
    bool Flag_2_on = shambase::is_flag_on<Flag2>(precondition_error);
    bool Flag_3_on = shambase::is_flag_on<Flag3>(precondition_error);

    REQUIRE(Flag_1_on);
    REQUIRE(Flag_2_on);
    REQUIRE(!Flag_3_on);
    REQUIRE(precondition_error == 3);
}

TestStart(Unittest, "shamalgs/atomic/ErrorCheckCounter", test_ErrorCheckCounter, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    shamalgs::atomic::ErrorCheckCounter error_util(sched, 4);

    sham::kernel_call(
        sched->get_queue(),
        sham::MultiRef{},
        sham::MultiRef{error_util},
        100,
        [](u32 i, auto error_util) {
            if (i == 2) {
                error_util.set_error(0);
            }
            if (i == 23 || i == 47) {
                error_util.set_error(1);
            }
            error_util.set_error(3);
        });

    auto precondition_error = error_util.get_outputs();

    REQUIRE_EQUAL(precondition_error.size(), 4);
    REQUIRE_EQUAL(precondition_error[0], 1);
    REQUIRE_EQUAL(precondition_error[1], 2);
    REQUIRE_EQUAL(precondition_error[2], 0);
    REQUIRE_EQUAL(precondition_error[3], 100);
}
