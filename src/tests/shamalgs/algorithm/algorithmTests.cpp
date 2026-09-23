// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/algorithm.hpp"
#include "shamalgs/primitives/sort_by_key_pow2_len.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "sortTests.hpp"

NEW_TEST(Unittest, "shamalgs/algorithm/sort_by_key_pow2_len", 1) {
    TestSortByKey<u32, u32> test(
        (TestSortByKey<u32, u32>::vFunctionCall) shamalgs::algorithm::sort_by_key_pow2_len);
    test.check();
}

NEW_TEST(Unittest, "shamalgs/algorithm/sort_by_key_pow2_len(usm)", 1) {
    TestSortByKeyUSM<u32, u32> test((TestSortByKeyUSM<u32, u32>::vFunctionCall)
                                        shamalgs::algorithm::sort_by_key_pow2_len<u32, u32>);

    if (!shamalgs::primitives::impl::is_impl_set_sort_by_key_pow2_len()) {
        shamalgs::primitives::impl::autoselect_impl_sort_by_key_pow2_len(
            shamsys::instance::get_compute_scheduler_ptr());
    }
    auto current_impl = shamalgs::primitives::impl::get_current_impl_sort_by_key_pow2_len();

    for (const std::string &impl :
         shamalgs::primitives::impl::get_default_impl_list_sort_by_key_pow2_len()) {
        shamalgs::primitives::impl::set_impl_sort_by_key_pow2_len(impl);
        shamlog_info_ln("tests", "testing implementation:", impl);
        test.check();
    }

    // reset to default
    shamalgs::primitives::impl::set_impl_sort_by_key_pow2_len(current_impl);
}

NEW_TEST(Benchmark, "shamalgs/algorithm/sort_by_key_pow2_len:benchmark", 1) {

    TestSortByKey<u32, u32> test(
        (TestSortByKey<u32, u32>::vFunctionCall) shamalgs::algorithm::sort_by_key_pow2_len);
    f64 rate = test.benchmark_one(1U << 24U);

    logger::raw_ln("rate =", rate);
}

NEW_TEST(Unittest, "shamalgs/algorithm/index_remap", 1) {
    TestIndexRemap<u32>(shamalgs::algorithm::index_remap<u32>).check();
}

NEW_TEST(Unittest, "shamalgs/algorithm/index_remap(usm)", 1) {
    TestIndexRemapUSM<u32>(shamalgs::algorithm::index_remap<u32>).check();
}
