// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/assert.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/assert", assert_testing, 1) {

    SHAM_ASSERT_NAMED("does it compile ?", true);

    sham::kernel_call(
        shamsys::instance::get_compute_scheduler().get_queue(),
        sham::MultiRef<>{},
        sham::MultiRef<>{},
        10,
        [](u32 i) {
            SHAM_ASSERT_NAMED("Assert in a SYCL kernel ?", true);
        });

    SHAM_ASSERT(true);
}
