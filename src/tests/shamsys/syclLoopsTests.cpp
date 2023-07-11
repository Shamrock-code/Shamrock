// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/integer.hpp"
#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include "shambase/sycl.hpp"
#include <vector>

namespace shambase {

    template<class LambdaKernel>
    inline void sycl_parralel_for(sycl::handler & cgh, u32 lenght, const u32 group_size, LambdaKernel && ker){

        constexpr u32 gsize = 8;
        u32 group_cnt = shambase::group_count(lenght, gsize);
        u32 len =group_cnt*gsize;

        cgh.parallel_for(sycl::nd_range<1>{len, gsize}, [=](sycl::nd_item<1> id){

            u64 gid = id.get_global_linear_id();
            if(gid >= lenght) return;

            ker(gid);
        });
    }
}

TestStart(Analysis, "sycl/loop_perfs", syclloopperfs, 1){

    std::vector<f64> speed_parfor ;

    auto fill_buf = [](u32 sz, sycl::buffer<f32> & buf){
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                acc[id] = id.get_linear_id();
            });

        }).wait();
    };

    f64 exp_test = 1.2;

    shambase::BenchmarkResult res_parfor = shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<f32> buf{sz};

        fill_buf(sz, buf);

        return shambase::timeit([&](){

            shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id){
                auto tmp = acc[id];
                acc[id] = tmp*tmp;
            });

        }).wait();

        },5);
    }, 10, 1e9, exp_test);


    shambase::BenchmarkResult res_ndrange = shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<f32> buf{sz};

        fill_buf(sz, buf);

        constexpr u32 gsize = 8;
        u32 group_cnt = shambase::group_count(sz, gsize);

        u32 len =group_cnt*gsize;

        return shambase::timeit([&](){

            shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};
            cgh.parallel_for(sycl::nd_range<1>{len, gsize}, [=](sycl::nd_item<1> id){

                u32 gid = id.get_global_linear_id();

                if(gid >= sz) return;

                auto tmp = acc[gid];
                acc[gid] = tmp*tmp;
            });

        }).wait();

        },5);
    }, 10, 1e9, exp_test);

    shambase::BenchmarkResult res_shampar = shambase::benchmark_pow_len([&](u32 sz){
        sycl::buffer<f32> buf{sz};

        fill_buf(sz, buf);

        constexpr u32 gsize = 8;
        u32 group_cnt = shambase::group_count(sz, gsize);

        u32 len =group_cnt*gsize;

        return shambase::timeit([&](){

            shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc {buf, cgh, sycl::read_write};

            shambase::sycl_parralel_for(cgh, sz, 256, [=](u64 gid){
                auto tmp = acc[gid];
                acc[gid] = tmp*tmp;
            });

        }).wait();

        },5);
    }, 10, 1e9, exp_test);

    PyScriptHandle hdnl{};

    hdnl.data()["x"] = res_parfor.counts;
    hdnl.data()["yparforbuf"] = res_parfor.times;
    hdnl.data()["yndrangeforbuf"] = res_ndrange.times;
    hdnl.data()["yshamrockpar"] = res_shampar.times;

    hdnl.exec(R"(
        import matplotlib.pyplot as plt
        import numpy as np

        X = np.array(x)

        Y = np.array(yparforbuf)
        plt.plot(X,Y/X,label = "parralel for (buffer) ")

        Y = np.array(yndrangeforbuf)
        plt.plot(X,Y/X,label = "ndrange for (buffer) ")

        Y = np.array(yshamrockpar)
        plt.plot(X,Y/X,label = "shamrock parralel for (buffer) ")

        plt.xlabel("s")
        plt.ylabel("N/t")

        plt.xscale('log')
        plt.yscale('log')

        plt.legend()

        plt.savefig("tests/figures/perfparfor.pdf")
    )");

}