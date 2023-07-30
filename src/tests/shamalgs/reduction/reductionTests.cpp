// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <map>
#include <random>
#include "shamalgs/details/reduction/details/fallbackReduction.hpp"
#include "shamalgs/reduction.hpp"
#include "shamalgs/details/reduction/details/sycl2020reduction.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/random/random.hpp"
#include "shamalgs/details/reduction/details/groupReduction.hpp"

using namespace shamalgs::random;

template<class T,class Fct> void unit_test_reduc_sum(std::string name, Fct && red_fct){
    std::vector<T> vals;

    constexpr u32 size_test = 1e6;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 100);


    for(u32 i = 0; i < size_test; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf {vals.data(),vals.size()};

        sycl_ret = red_fct(shamsys::instance::get_compute_queue(), buf, 0, size_test);

    }

    {
        check_val = T{0};
        for(auto & f : vals){
            check_val += f;
        }
    }

    T delt = (sycl_ret - check_val)/1e8;
    auto dot = shambase::sycl_utils::g_sycl_dot(delt, delt);

    shamtest::asserts().assert_float_equal(name, dot,0,1e-9);

}




void unit_test_reduc_sum(){

    unit_test_reduc_sum<f64>("reduction : main (f64)",
        [](sycl::queue & q, sycl::buffer<f64> & buf1, u32 start_id, u32 end_id) -> f64 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        }
    );

    unit_test_reduc_sum<f32>("reduction : main (f32)",
        [](sycl::queue & q, sycl::buffer<f32> & buf1, u32 start_id, u32 end_id) -> f32 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        }
    );

    unit_test_reduc_sum<u32>("reduction : main (u32)",
        [](sycl::queue & q, sycl::buffer<u32> & buf1, u32 start_id, u32 end_id) -> u32 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        }
    );

    unit_test_reduc_sum<f64_3>("reduction : main (f64_3)",
        [](sycl::queue & q, sycl::buffer<f64_3> & buf1, u32 start_id, u32 end_id) -> f64_3 {
            return shamalgs::reduction::sum(q, buf1, start_id, end_id);
        }
    );

}

TestStart(Unittest, "shamalgs/reduction/sum", reduc_kernel_utest, 1){
    unit_test_reduc_sum();
}





//////////////////////////////////////:
// benchmarks
//////////////////////////////////////:

TestStart(Benchmark, "shamalgs/reduction/sum", benchmark_reductionkernels, 1){

    std::map<std::string, shambase::BenchmarkResult> results;

    using T = f64;

    f64 exp_test = 1.2;

    results.emplace("fallback",shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

        //do op on GPU to force locality on GPU before test
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = acc[id]*acc[id];
            });

        }).wait();

        return shambase::timeit([&](){

            T sum = shamalgs::reduction::details::FallbackReduction<T>::sum(shamsys::instance::get_compute_queue(),buf,0, sz);
            shamsys::instance::get_compute_queue().wait();

        });
    }, 10, 1e8, exp_test));

    results.emplace("sycl2020",shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

        //do op on GPU to force locality on GPU before test
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = acc[id]*acc[id];
            });

        }).wait();

        return shambase::timeit([&](){

            T sum = shamalgs::reduction::details::SYCL2020<T>::sum(shamsys::instance::get_compute_queue(),buf,0, sz);
            shamsys::instance::get_compute_queue().wait();

        });
    }, 10, 1e8, exp_test));


    results.emplace("slicegroup8",shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

        //do op on GPU to force locality on GPU before test
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = acc[id]*acc[id];
            });

        }).wait();

        return shambase::timeit([&](){

            T sum = shamalgs::reduction::details::GroupReduction<T,8>::sum(shamsys::instance::get_compute_queue(),buf,0, sz);
            shamsys::instance::get_compute_queue().wait();

        });
    }, 10, 1e8, exp_test));

    results.emplace("slicegroup32",shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

        //do op on GPU to force locality on GPU before test
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = acc[id]*acc[id];
            });

        }).wait();

        return shambase::timeit([&](){

            T sum = shamalgs::reduction::details::GroupReduction<T,32>::sum(shamsys::instance::get_compute_queue(),buf,0, sz);
            shamsys::instance::get_compute_queue().wait();

        });
    }, 10, 1e8, exp_test));


    results.emplace("slicegroup128",shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<T> buf = shamalgs::random::mock_buffer<T>(0x111, sz);

        //do op on GPU to force locality on GPU before test
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = acc[id]*acc[id];
            });

        }).wait();

        return shambase::timeit([&](){

            T sum = shamalgs::reduction::details::GroupReduction<T,128>::sum(shamsys::instance::get_compute_queue(),buf,0, sz);
            shamsys::instance::get_compute_queue().wait();

        });
    }, 10, 1e8, exp_test));

    PyScriptHandle hdnl{};

    for(auto & [key,res] : results){
        hdnl.data()["x"] = res.counts;
        hdnl.data()[key.c_str()] = res.times;
    }

    hdnl.exec(R"(
        import matplotlib.pyplot as plt
        import numpy as np

        X = np.array(x)

        Y = np.array(fallback)
        plt.plot(X,Y/X,label = "fallback")

        Y = np.array(sycl2020)
        plt.plot(X,Y/X,label = "sycl2020")

        Y = np.array(slicegroup8)
        plt.plot(X,Y/X,label = "slicegroup8")

        Y = np.array(slicegroup32)
        plt.plot(X,Y/X,label = "slicegroup32")

        Y = np.array(slicegroup128)
        plt.plot(X,Y/X,label = "slicegroup128")

        plt.xlabel("s")
        plt.ylabel("N/t")

        plt.xscale('log')
        plt.yscale('log')

        plt.legend()

        plt.savefig("tests/figures/shamalgsreduc.pdf")
    )");

}