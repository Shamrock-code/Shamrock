// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/make_ndrange.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

namespace {
    /// Checks that the ndrange returned by `sham::make_ndrange` has the expected global and
    /// local sizes for the given `wg_size` / `nthread` inputs.
    void check_ndrange(u32 wg_size, u32 nthread, u32 expected_global) {
        sycl::nd_range<1> ndr = sham::make_ndrange(wg_size, nthread);

        REQUIRE_EQUAL(ndr.get_global_range().size(), expected_global);
        REQUIRE_EQUAL(ndr.get_local_range().size(), wg_size);
        REQUIRE_EQUAL(ndr.get_group_range().size(), expected_global / wg_size);
    }
} // namespace

NEW_TEST(Unittest, "shambackends/make_ndrange.hpp:make_ndrange", 1) {

    // nthread already a multiple of wg_size -> no rounding needed
    check_ndrange(32, 64, 64);
    check_ndrange(32, 32, 32);
    check_ndrange(16, 160, 160);

    // nthread one more than a multiple of wg_size -> rounds up to the next block
    check_ndrange(32, 65, 96);
    check_ndrange(16, 17, 32);

    // nthread one less than a multiple of wg_size -> rounds up to that multiple
    check_ndrange(32, 63, 64);
    check_ndrange(16, 15, 16);

    // nthread smaller than wg_size -> rounds up to a single full work-group
    check_ndrange(32, 1, 32);
    check_ndrange(1024, 5, 1024);

    // nthread == 1 with wg_size == 1 -> trivial single-thread launch
    check_ndrange(1, 1, 1);

    // wg_size == 1 -> no rounding ever occurs, regardless of nthread
    check_ndrange(1, 7, 7);
    check_ndrange(1, 1000, 1000);

    // large values, well within u32 range, no overflow in the rounding arithmetic
    check_ndrange(256, 1'000'000, 1'000'192);
    check_ndrange(128, 1'048'576, 1'048'576);
}
