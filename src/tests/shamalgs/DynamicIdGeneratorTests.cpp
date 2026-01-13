// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/integer.hpp"
#include "shambase/time.hpp"
#include "shamalgs/atomic/DynamicIdGenerator.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/atomic/DynamicIdGenerator", test_dynamicidgen, 1) {

    constexpr u32 gsize = 64;

    u32 group_cnt = shambase::group_count(1e6, gsize);

    u32 len = group_cnt * gsize;
    sycl::buffer<i32> ret_buf(len);

    shamalgs::atomic::DynamicIdGenerator<i32, gsize> id_gen(shamsys::instance::get_compute_queue());

    shamsys::instance::get_compute_queue().submit([&, group_cnt, len](sycl::handler &cgh) {
        auto dyn_id = id_gen.get_access(cgh);

        sycl::accessor acc{ret_buf, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::nd_range<1>{len, gsize}, [=](sycl::nd_item<1> id) {
            shamalgs::atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

            acc[group_id.dyn_global_id] = group_id.dyn_global_id;
        });
    });

    shamsys::instance::get_compute_queue().wait();

    // shamalgs::memory::print_buf(ret_buf, len, 16);
}

template<u32 gsize>
f64 bench_one_idgen(u32 wanted_len) {

    f64 cnt = 5;

    if (wanted_len < 2e6) {
        cnt = 20;
    } else if (wanted_len < 1e5) {
        cnt = 100;
    } else if (wanted_len < 1e4) {
        cnt = 1000;
    }

    u32 group_cnt = shambase::group_count(wanted_len, gsize);

    u32 len = group_cnt * gsize;
    sycl::buffer<i32> ret_buf(len);

    shamsys::instance::get_compute_queue().wait();

    return shambase::timeit(
        [&]() {
            shamalgs::atomic::DynamicIdGenerator<i32, gsize> id_gen(
                shamsys::instance::get_compute_queue());

            shamsys::instance::get_compute_queue().submit([&, group_cnt, len](sycl::handler &cgh) {
                auto dyn_id = id_gen.get_access(cgh);

                sycl::accessor acc{ret_buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::nd_range<1>{len, gsize}, [=](sycl::nd_item<1> id) {
                    shamalgs::atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                    acc[group_id.dyn_global_id] = group_id.dyn_global_id;
                });
            });

            shamsys::instance::get_compute_queue().wait();
        },
        cnt);
}

TestStart(Benchmark, "shamalgs/atomic/DynamicIdGenerator:benchmark", bench_dynamicidgen, 1) {

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<4>(cnt);
            },
            1e3,
            1e8,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=4");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<8>(cnt);
            },
            1e3,
            1e8,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=8");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<16>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=16");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<32>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=32");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<64>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=64");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<128>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=128");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<256>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=256");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }

    {
        auto res = shambase::benchmark_pow_len(
            [](u32 cnt) -> f64 {
                return bench_one_idgen<512>(cnt);
            },
            1e3,
            1e9,
            1.1);

        auto &dat_test = shamtest::test_data().new_dataset("dynamic id gen gsize=512");

        dat_test.add_data("Nobj", res.counts);
        dat_test.add_data("time", res.times);
    }
}
