// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shambackends/math.hpp"

TestStart(Unittest, "shambackends/math.hpp:roundup_pow2_clz", shambackendsmathroundup_pow2_clz, 1){

    _AssertEqual(sham::roundup_pow2_clz<u32>(0) , 0)
    _AssertEqual(sham::roundup_pow2_clz<u32>(1) , 1)
    _AssertEqual(sham::roundup_pow2_clz<u32>(2) , 2)
    _AssertEqual(sham::roundup_pow2_clz<u32>(3) , 4)


    _AssertEqual(sham::roundup_pow2_clz<u32>(2147483647) , 2147483648)
    _AssertEqual(sham::roundup_pow2_clz<u32>(2147483648) , 2147483648)
    _AssertEqual(sham::roundup_pow2_clz<u32>(2147483649) , 0)
    _AssertEqual(sham::roundup_pow2_clz<u32>(4294967295) , 0)

}