// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include "shambase/sycl.hpp"
#include <hipSYCL/sycl/buffer.hpp>
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <vector>



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
    }, 10, 1e9, 1.1);

    PyScriptHandle hdnl{};

    hdnl.data()["x"] = res_parfor.counts;
    hdnl.data()["y"] = res_parfor.times;

    hdnl.exec(R"(
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.savefig("tests/figures/perfparfor.pdf")
    )");

}