// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/key_morton_sort.hpp"
#include <random>

template<class u_morton, SortImplType impl>
void unit_test_key_pair() {
    std::vector<u_morton> morton_list;

    constexpr u32 size_test = 16;

    for (u32 i = 0; i < size_test; i++) {
        morton_list.push_back(i);
    }

    std::mt19937 eng(0x1111);

    shuffle(morton_list.begin(), morton_list.end(), eng);

    std::vector<u_morton> unsorted(morton_list.size());

    std::copy(morton_list.begin(), morton_list.end(), unsorted.begin());

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton
            = std::make_unique<sycl::buffer<u_morton>>(morton_list.data(), morton_list.size());
        std::unique_ptr<sycl::buffer<u32>> buf_index
            = std::make_unique<sycl::buffer<u32>>(morton_list.size());

        sycl_sort_morton_key_pair<u_morton, impl>(
            shamsys::instance::get_compute_queue(), size_test, buf_index, buf_morton);
    }

    std::sort(unsorted.begin(), unsorted.end());

    for (u32 i = 0; i < size_test; i++) {
        REQUIRE_EQUAL_NAMED(
            "index [" + shambase::format_printf("%d", i) + "]", unsorted[i], morton_list[i]);
    }
}

template<class u_morton, SortImplType impl>
f64 benchmark_key_pair_sort(const u32 &nobj) {
    std::vector<u_morton> morton_list;

    for (u32 i = 0; i < nobj; i++) {
        morton_list.push_back(i);
    }

    std::mt19937 eng(0x1111);

    shuffle(morton_list.begin(), morton_list.end(), eng);

    shambase::Timer t;

    {
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton
            = std::make_unique<sycl::buffer<u_morton>>(morton_list.data(), morton_list.size());
        std::unique_ptr<sycl::buffer<u32>> buf_index
            = std::make_unique<sycl::buffer<u32>>(morton_list.size());

        shamsys::instance::get_compute_queue().wait();

        t.start();

        sycl_sort_morton_key_pair<u_morton, impl>(
            shamsys::instance::get_compute_queue(), nobj, buf_index, buf_morton);

        shamsys::instance::get_compute_queue().wait();

        t.end();
    }

    return t.nanosec * 1e-9;
}

TestStart(Unittest, "core/tree/kernels/key_pair_sort", key_pair_sort_test, 1) {
    unit_test_key_pair<u32, MultiKernel>();
    unit_test_key_pair<u64, MultiKernel>();
}

constexpr u32 lim_bench = 1e9;

template<class u_morton, SortImplType impl>
void wrapper_bench_key_sort(std::string name) {

    logger::info_ln("ShamrockTest", "testing :", name);

    std::vector<f64> test_sz;
    for (f64 i = 16; i < lim_bench; i *= 2) {
        test_sz.push_back(i);
    }

    auto &res = shamtest::test_data().new_dataset(name);

    std::vector<f64> results;

    for (const f64 &sz : test_sz) {
        shamlog_debug_ln("ShamrockTest", "N=", sz);
        results.push_back(benchmark_key_pair_sort<u_morton, impl>(sz));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);
}

TestStart(Benchmark, "core/tree/kernels/key_pair_sort (benchmark)", key_pair_sort_bench, 1) {

    wrapper_bench_key_sort<u32, MultiKernel>("bitonic u32 multi kernel");
    wrapper_bench_key_sort<u32, MultiKernel>("bitonic u64 multi kernel");
}
