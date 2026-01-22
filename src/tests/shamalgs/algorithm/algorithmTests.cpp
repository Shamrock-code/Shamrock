// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/algorithm.hpp"
#include "sortTests.hpp"

TestStart(Unittest, "shamalgs/algorithm/sort_by_key", test_sort_by_key_func, 1) {
    TestSortByKey<u32, u32> test(
        (TestSortByKey<u32, u32>::vFunctionCall) shamalgs::algorithm::sort_by_key);
    test.check();
}

TestStart(Unittest, "shamalgs/algorithm/sort_by_key(usm)", test_sort_by_key_func_usm, 1) {
    TestSortByKeyUSM<u32, u32> test(
        (TestSortByKeyUSM<u32, u32>::vFunctionCall) shamalgs::algorithm::sort_by_key<u32, u32>);
    test.check();
}

TestStart(
    Benchmark, "shamalgs/algorithm/sort_by_key:benchmark", test_sort_by_key_func_benchmark, 1) {

    TestSortByKey<u32, u32> test(
        (TestSortByKey<u32, u32>::vFunctionCall) shamalgs::algorithm::sort_by_key);
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}

TestStart(Unittest, "shamalgs/algorithm/index_remap", test_index_remap_func, 1) {
    TestIndexRemap<u32>(shamalgs::algorithm::index_remap<u32>).check();
}

TestStart(Unittest, "shamalgs/algorithm/index_remap(usm)", test_index_remap_func_usm, 1) {
    TestIndexRemapUSM<u32>(shamalgs::algorithm::index_remap<u32>).check();
}
