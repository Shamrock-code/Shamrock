// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "numericTests.hpp"
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

template<class T>
struct TestExclScan {

    using vFunctionCall = sycl::buffer<T> (*)(sycl::queue &, sycl::buffer<T> &, u32);

    vFunctionCall fct;

    explicit TestExclScan(vFunctionCall arg) : fct(arg) {};

    void check() {
        if constexpr (std::is_same<u32, T>::value) {

            u32 len_test = 1e5;

            std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(0x111, len_test, 0, 10);

            std::vector<u32> data_buf(data);

            std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);

            sycl::buffer<u32> buf{data_buf.data(), data_buf.size()};

            // shamalgs::memory::print_buf(buf, 4096, 16, "{:4} ");

            sycl::buffer<u32> res
                = fct(shamsys::instance::get_compute_queue(), buf, data_buf.size());

            // shamalgs::memory::print_buf(res, len_test, 16, "{:4} ");

            {
                sycl::host_accessor acc{res, sycl::read_only};

                std::vector<u32> result;
                for (u32 i = 0; i < data_buf.size(); i++) {
                    result.push_back(acc[i]);
                }

                REQUIRE_EQUAL_NAMED("inclusive scan match", result, data);
            }

            // sycl::buffer<u32> tmp (data.data(), data.size());
            // shamalgs::memory::print_buf(tmp, len_test, 16, "{:4} ");
        }
    }

    f64 benchmark_one(u32 len) {

        sycl::buffer<u32> buf = shamalgs::random::mock_buffer<u32>(0x111, len, 0, 50);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        shamalgs::memory::move_buffer_on_queue(q, buf);

        q.wait();

        shambase::Timer t;
        t.start();
        sycl::buffer<u32> res = fct(q, buf, len);

        q.wait();
        t.end();

        return (t.nanosec * 1e-9);
    }

    f64 bench_one_avg(u32 len) {
        f64 sum = 0;

        f64 cnt = 4;

        if (len < 2e6) {
            cnt = 10;
        } else if (len < 1e5) {
            cnt = 100;
        } else if (len < 1e4) {
            cnt = 1000;
        }

        for (u32 i = 0; i < cnt; i++) {
            sum += benchmark_one(len);
        }

        return sum / cnt;
    }

    py::dict benchmark(u32 lim_bench = 1e8) {

        std::vector<u32> sizes;
        std::vector<f64> times;

        logger::info_ln("TestExclScan", "testing :", __PRETTY_FUNCTION__);

        f64 i = 1e3;
        while (i < lim_bench) {
            sizes.push_back(u32(i));
            i = i * 1.1_f64;
        }

        for (const u32 &sz : sizes) {
            shamlog_debug_ln("ShamrockTest", "N=", sz);
            times.push_back(bench_one_avg(sz));
        }

        py::dict ret;
        ret["sizes"] = sizes;
        ret["times"] = times;

        return ret;
    }
};

template<class T>
struct TestExclScanUSM {

    using vFunctionCall
        = sham::DeviceBuffer<T> (*)(sham::DeviceScheduler_ptr, sham::DeviceBuffer<T> &, u32);

    vFunctionCall fct;

    explicit TestExclScanUSM(vFunctionCall arg) : fct(arg) {};

    void check() {
        if constexpr (std::is_same<u32, T>::value) {

            u32 len_test = 1e5;

            std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(0x111, len_test, 0, 10);

            std::vector<u32> data_buf(data);

            std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);

            sham::DeviceBuffer<u32> buf{
                data_buf.size(), shamsys::instance::get_compute_scheduler_ptr()};
            buf.copy_from_stdvec(data_buf);

            REQUIRE_EQUAL(buf.get_size(), len_test);

            sham::DeviceBuffer<u32> res
                = fct(shamsys::instance::get_compute_scheduler_ptr(), buf, data_buf.size());

            REQUIRE_EQUAL(res.get_size(), len_test);

            // shamalgs::memory::print_buf(res, len_test, 16, "{:4} ");

            {
                std::vector<u32> result = res.copy_to_stdvec();

                REQUIRE_EQUAL(result.size(), data_buf.size());

                REQUIRE_EQUAL_NAMED("exclusive scan match", result, data);
            }

            // sycl::buffer<u32> tmp (data.data(), data.size());
            // shamalgs::memory::print_buf(tmp, len_test, 16, "{:4} ");
        }
    }

    f64 benchmark_one(u32 len) {

        std::vector<u32> data_buf = shamalgs::primitives::mock_vector<u32>(0x111, len, 0, 50);

        auto dev_ptr = shamsys::instance::get_compute_scheduler_ptr();

        sham::DeviceBuffer<u32> buf{data_buf.size(), dev_ptr};
        buf.copy_from_stdvec(data_buf);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        {
            sham::EventList depends_list;
            u32 *res_ptr = buf.get_write_access(depends_list);
            depends_list.wait_and_throw();
            buf.complete_event_state(sycl::event{});
        }
        q.wait();

        shambase::Timer t;
        t.start();
        sham::DeviceBuffer<u32> res = fct(dev_ptr, buf, len);

        {
            sham::EventList depends_list;
            u32 *res_ptr = res.get_write_access(depends_list);
            depends_list.wait_and_throw();
            res.complete_event_state(sycl::event{});
        }
        t.end();

        return (t.nanosec * 1e-9);
    }

    f64 bench_one_avg(u32 len) {
        f64 sum = 0;

        f64 cnt = 4;

        if (len < 2e6) {
            cnt = 10;
        } else if (len < 1e5) {
            cnt = 100;
        } else if (len < 1e4) {
            cnt = 1000;
        }

        for (u32 i = 0; i < cnt; i++) {
            sum += benchmark_one(len);
        }

        return sum / cnt;
    }

    py::dict benchmark(u32 lim_bench = 1e8) {

        std::vector<u32> sizes;
        std::vector<f64> times;

        logger::info_ln("TestExclScan", "testing :", __PRETTY_FUNCTION__);

        f64 i = 1e3;
        while (i < lim_bench) {
            sizes.push_back(u32(i));
            i *= 1.1_f64;
        }

        for (const u32 &sz : sizes) {
            shamlog_debug_ln("ShamrockTest", "N=", sz);
            times.push_back(bench_one_avg(sz));
        }

        py::dict ret;
        ret["sizes"] = sizes;
        ret["times"] = times;

        return ret;
    }
};

template<class T>
struct TestExclScanInplace {

    using vFunctionCall = void (*)(sycl::queue &, sycl::buffer<T> &, u32);

    vFunctionCall fct;

    explicit TestExclScanInplace(vFunctionCall arg) : fct(arg) {};

    void check() {
        if constexpr (std::is_same<u32, T>::value) {

            u32 len_test = 1e5;

            std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(0x111, len_test, 0, 10);

            std::vector<u32> data_buf(data);

            std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);

            sycl::buffer<u32> buf = shamalgs::memory::vec_to_buf(data_buf);

            // shamalgs::memory::print_buf(buf, 4096, 16, "{:4} ");

            fct(shamsys::instance::get_compute_queue(), buf, data_buf.size());

            // shamalgs::memory::print_buf(res, len_test, 16, "{:4} ");

            {
                sycl::host_accessor acc{buf, sycl::read_only};

                std::vector<u32> result;
                for (u32 i = 0; i < data_buf.size(); i++) {
                    result.push_back(acc[i]);
                }

                REQUIRE_EQUAL_NAMED("inclusive scan match", result, data);
            }
        }
    }

    f64 benchmark_one(u32 len) {

        sycl::buffer<u32> buf = shamalgs::random::mock_buffer<u32>(0x111, len, 0, 50);

        sycl::queue &q = shamsys::instance::get_compute_queue();

        shamalgs::memory::move_buffer_on_queue(q, buf);

        q.wait();

        shambase::Timer t;
        t.start();
        fct(q, buf, len);

        q.wait();
        t.end();

        return (t.nanosec * 1e-9);
    }

    f64 bench_one_avg(u32 len) {
        f64 sum = 0;

        f64 cnt = 4;

        if (len < 2e6) {
            cnt = 10;
        } else if (len < 1e5) {
            cnt = 100;
        } else if (len < 1e4) {
            cnt = 1000;
        }

        for (u32 i = 0; i < cnt; i++) {
            sum += benchmark_one(len);
        }

        return sum / cnt;
    }

    struct BenchRes {
        std::vector<f64> sizes;
        std::vector<f64> times;
    };

    BenchRes benchmark(u32 lim_bench = 1e8) {
        BenchRes ret;

        logger::info_ln("TestExclScanInplace", "testing :", __PRETTY_FUNCTION__);

        for (f64 i = 1e3; i < lim_bench; i *= 1.1) {
            ret.sizes.push_back(i);
        }

        for (const f64 &sz : ret.sizes) {
            shamlog_debug_ln("ShamrockTest", "N=", sz);
            ret.times.push_back(bench_one_avg(sz));
        }

        return ret;
    }
};

TestStart(
    Unittest,
    "shamalgs/numeric/details/exclusive_sum_in_place_fallback",
    exclsumtest_fallback_inplace,
    1) {

    TestExclScanInplace<u32> test((TestExclScanInplace<u32>::vFunctionCall)
                                      shamalgs::numeric::details::exclusive_sum_in_place_fallback);
    test.check();
}

/*
TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_in_place_atomic_decoupled_v5",
test_exclusive_sum_in_place_atomic_decoupled_v5, 1){

    TestExclScanInplace<u32> test
((TestExclScanInplace<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_in_place_atomic_decoupled_v5<u32,
256>); test.check();
}
*/

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_fallback", exclsumtest_fallback, 1) {

    TestExclScan<u32> test(
        (TestExclScan<u32>::vFunctionCall) shamalgs::numeric::details::exclusive_sum_fallback);
    test.check();
}

TestStart(
    Unittest,
    "shamalgs/numeric/details/exclusive_sum_gpugems39",
    test_exclusive_sum_gpugems39_1,
    1) {

    TestExclScan<u32> test(
        (TestExclScan<u32>::vFunctionCall) shamalgs::numeric::details::exclusive_sum_gpugems39_1);
    test.check();
}
/*
TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic2pass",
test_exclusive_sum_atomic2pass, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic2pass<u32,16>);
    test.check();

}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic2pass_v2",
test_exclusive_sum_atomic2pass_v2, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic2pass_v2<u32,16>);
    test.check();

}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled",
test_exclusive_sum_atomic_decoupled, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v2",
test_exclusive_sum_atomic_decoupled_v2, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,16>);
    test.check();

}

TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v3",
test_exclusive_sum_atomic_decoupled_v3, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,512>);
    test.check();

}


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v4",
test_exclusive_sum_atomic_decoupled_v4, 1){

    TestExclScan<u32> test
((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v4<u32,512>);
    test.check();

}
*/
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
TestStart(
    Unittest,
    "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v5",
    test_exclusive_sum_atomic_decoupled_v5,
    1) {

    TestExclScan<u32> test(
        (TestExclScan<u32>::vFunctionCall)
            shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5<u32, 512>);
    test.check();
}
#endif

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
TestStart(
    Unittest,
    "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v5_usm",
    test_exclusive_sum_atomic_decoupled_v5_usm,
    1) {

    TestExclScanUSM<u32> test(
        (TestExclScanUSM<u32>::vFunctionCall)
            shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5_usm<u32, 512>);
    test.check();
}
#endif

// TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_atomic_decoupled_v6",
// test_exclusive_sum_atomic_decoupled_v6, 1){
//
//     TestExclScan<u32> test
//     ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32,64,8>);
//     test.check();
//
// }

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
TestStart(
    Unittest,
    "shamalgs/numeric/details/exclusive_sum_sycl_jointalg",
    test_exclusive_sum_sycl_jointalg,
    1) {

    TestExclScan<u32> test((TestExclScan<u32>::vFunctionCall)
                               shamalgs::numeric::details::exclusive_sum_sycl_jointalg<u32, 32>);
    test.check();
}
#endif

TestStart(Benchmark, "shamalgs/numeric/details/exclusive_sum:benchmark", bench_exclusive_sum, 1) {
    py::dict results;
    /*
        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::exclusive_sum); auto result =
       test.benchmark();

            auto & res = shamtest::test_data().new_dataset("public u32");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_1);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("gpugems39 v1 u32");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }
    */
    {
        TestExclScan<u32> test((TestExclScan<u32>::vFunctionCall)
                                   shamalgs::numeric::details::exclusive_sum_gpugems39_2);
        results["exclusive_sum_gpugems39_2"] = test.benchmark();
    }

    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall) shamalgs::numeric::details::exclusive_sum_fallback);
        results["exclusive_sum_fallback"] = test.benchmark();
    }

    /*
        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,256>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize = 256");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,512>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize = 512");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled<u32,1024>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled u32 gsize =
       1024");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,64>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize =
       64");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,128>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize =
       128");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,256>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize =
       256");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v2<u32,512>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v2 u32 gsize =
       512");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        } */

    /*
        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,256>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize =
       256");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,512>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize =
       512");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v3<u32,1024>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v3 u32 gsize =
       1024");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }


        {
            TestExclScan<u32> test
       ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v4<u32,512>);
            auto result = test.benchmark();

            auto & res = shamtest::test_data().new_dataset("atomic scan decoupled v4 u32 gsize =
       512");

            res.add_data("Nobj", result.sizes);
            res.add_data("t_sort", result.times);
        }

    */
    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5<u32, 512>);
        results["D. Merrill u32 gsize = 512"] = test.benchmark();
    }
    {
        TestExclScanUSM<u32> test(
            (TestExclScanUSM<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5_usm<u32, 512>);
        results["D. Merrill u32 gsize = 512 (USM)"] = test.benchmark();
    }

    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_sycl_jointalg<u32, 512>);
        results["sycl joint excl sum u32 gsize = 512"] = test.benchmark(4e6);
    }

    /*
    {
        TestExclScan<u32> test
    ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32,1024,512>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("D. Merrill parr. u32 gsize = 1024/512");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    {
        TestExclScan<u32> test
    ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32,1024,1024>);
        auto result = test.benchmark();

        auto & res = shamtest::test_data().new_dataset("D. Merrill parr. u32 gsize = 1024/1024");

        res.add_data("Nobj", result.sizes);
        res.add_data("t_sort", result.times);
    }
    */

    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32, 512, 256>);
        results["D. Merrill parr. u32 gsize = 512/256"] = test.benchmark();
    }
    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32, 512, 512>);
        results["D. Merrill parr. u32 gsize = 512/512"] = test.benchmark();
    }

    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32, 256, 128>);
        results["D. Merrill parr. u32 gsize = 256/128"] = test.benchmark();
    }
    {
        TestExclScan<u32> test(
            (TestExclScan<u32>::vFunctionCall)
                shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v6<u32, 256, 256>);
        results["D. Merrill parr. u32 gsize = 256/256"] = test.benchmark();
    }

    PyScriptHandle hdnl{};

    hdnl.data()["results"] = results;

    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')


        for name, result in results.items():
            plt.plot(result["sizes"], np.array(result["times"])/np.array(result["sizes"]), label=name)

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(r"$N$")
        plt.ylabel(r"$t_{\text{scan}}/N$")
        plt.legend()

        plt.tight_layout()

        plt.savefig("tests/figures/excl_scan_perf.pdf", dpi = 300)

    )");
}
