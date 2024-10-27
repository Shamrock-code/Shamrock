// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambackends/math.hpp:roundup_pow2_clz", shambackendsmathroundup_pow2_clz, 1) {

    _AssertEqual(sham::roundup_pow2_clz<u32>(0), 1) _AssertEqual(sham::roundup_pow2_clz<u32>(1), 1)
        _AssertEqual(sham::roundup_pow2_clz<u32>(2), 2)
            _AssertEqual(sham::roundup_pow2_clz<u32>(3), 4)

                _AssertEqual(sham::roundup_pow2_clz<u32>(2147483647), 2147483648)
                    _AssertEqual(sham::roundup_pow2_clz<u32>(2147483648), 2147483648)
                        _AssertEqual(sham::roundup_pow2_clz<u32>(2147483649), 0)
                            _AssertEqual(sham::roundup_pow2_clz<u32>(4294967295), 0)
}
