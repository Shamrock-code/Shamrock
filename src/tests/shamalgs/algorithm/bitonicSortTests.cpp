// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/algorithm/bitonicSort.hpp"
#include "shamalgs/details/algorithm/bitonicSort_updated_usm.hpp"
#include "shamalgs/details/algorithm/radixSortOnesweep.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "sortTests.hpp"

TestStart(Unittest, "shamalgs/algorithm/details/bitonicSort_legacy", test_bitonic_sort_legacy, 1) {

    TestSortByKey<u32, u32> test((TestSortByKey<u32, u32>::vFunctionCall)
                                     shamalgs::algorithm::details::sort_by_key_bitonic_legacy);
    test.check();
}

TestStart(
    Unittest, "shamalgs/algorithm/details/bitonicSort_updated", test_bitonic_sort_updated, 1) {

    TestSortByKey<u32, u32> test(
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32, u32, 16>);
    test.check();
}

TestStart(
    Unittest,
    "shamalgs/algorithm/details/bitonicSort_updated_usm",
    test_bitonic_sort_updated_usm,
    1) {

    TestSortByKeyUSM<u32, u32> test(
        (TestSortByKeyUSM<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_usm<u32, u32, 16>);
    test.check();
}

TestStart(
    Unittest, "shamalgs/algorithm/details/bitonicSort_fallback", test_bitonic_sort_fallback, 1) {

    TestSortByKey<u32, u32> test((TestSortByKey<u32, u32>::vFunctionCall)
                                     shamalgs::algorithm::details::sort_by_key_bitonic_fallback);
    test.check();
}

/*
TestStart(Unittest, "shamalgs/algorithm/details/sort_by_key_radix_onesweep_v3",
test_sort_by_key_radix_onesweep_v3, 1){

    TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,16,2>
        );
    test.check();
}
*/

#if false
TestStart(Benchmark, "shamalgs/algorithm/details/bitonicSorts:benchmark",
    test_bitonic_sort_legacy_benchmark, 1){

    /*
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_legacy
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic legacy (u32, u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,8>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,8)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */


    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,16>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,16)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    /*
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32,u32,32>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated (u32,u32,32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,1>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,1");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,2>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,2");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,4>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,4");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_radix_onesweep<u32,u32,512,8>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("A. Adinets et al. 2022 rsort g512,8");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    /* disabled
    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,8>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,8)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }


    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,16>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,16)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_updated_xor_swap<u32,u32,32>
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic updated xor swap (u32,u32,32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    {
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic fallback (u32,u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }

    if(false){
        TestSortByKey<u32, u32>test (
        (TestSortByKey<u32, u32>::vFunctionCall)
            shamalgs::algorithm::sort_by_key
        );

        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("bitonic public (u32,u32)");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }


}
#endif

TestStart(Benchmark, "shamalgs/algorithm/details/bitonicSorts:benchmark", test_bitonic_sort, 1) {

    PyScriptHandle hdnl{};

    {
        TestSortByKey<u32, u32> test(
            (TestSortByKey<u32, u32>::vFunctionCall)
                shamalgs::algorithm::details::sort_by_key_bitonic_fallback);

        auto result = test.benchmark();

        hdnl.data()["label_1"]  = "bitonic fallback (u32,u32)";
        hdnl.data()["Nobj_1"]   = result.sizes;
        hdnl.data()["t_sort_1"] = result.times;
    }

    {
        TestSortByKey<u32, u32> test(
            (TestSortByKey<u32, u32>::vFunctionCall)
                shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32, u32, 16>);

        auto result = test.benchmark();

        hdnl.data()["label_2"]  = "bitonic updated (u32,u32,16)";
        hdnl.data()["Nobj_2"]   = result.sizes;
        hdnl.data()["t_sort_2"] = result.times;
    }

    {
        TestSortByKey<u32, u32> test(
            (TestSortByKey<u32, u32>::vFunctionCall)
                shamalgs::algorithm::details::sort_by_key_bitonic_updated<u32, u32, 32>);

        auto result = test.benchmark();

        hdnl.data()["label_3"]  = "bitonic updated (u32,u32,32)";
        hdnl.data()["Nobj_3"]   = result.sizes;
        hdnl.data()["t_sort_3"] = result.times;
    }

    hdnl.exec(R"py(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        plt.plot(Nobj_1, np.array(t_sort_1)/np.array(Nobj_1), label=label_1)
        plt.plot(Nobj_2, np.array(t_sort_2)/np.array(Nobj_2), label=label_2)
        plt.plot(Nobj_3, np.array(t_sort_3)/np.array(Nobj_3), label=label_3)


        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(r"$N$")

        plt.ylabel(r"$t_{\rm sort}/N$ (s)")
        plt.legend()

        plt.tight_layout()

        plt.savefig("tests/figures/sort_benchmark.pdf")
    )py");

    TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{figures/sort_benchmark.pdf}
        \caption{Bencmark of the sorting algs}
        \end{figure}

    )==")
}
