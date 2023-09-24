// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/details/random/random.hpp"
#include "shambase/Constants.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambase/time.hpp"
#include "shammath/derivatives.hpp"
#include "shammath/integrator.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

#include "shammath/sphkernels.hpp"
#include <vector>



template<class Ker>
inline void validate_kernel_3d(typename Ker::Tscal tol,typename Ker::Tscal dx,typename Ker::Tscal dx_int){

    using Tscal = typename Ker::Tscal;

    // test finite support
    _AssertEqual(Ker::f(Ker::Rkern) , 0);
    _AssertEqual(Ker::W_3d(Ker::Rkern,1) , 0);

    Tscal gen_norm3d = Ker::Generator::norm_3d;

    // test f <-> W scale relations
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern) , Ker::W_3d(Ker::Rkern,1));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/2) , Ker::W_3d(Ker::Rkern/2,1));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/3) , Ker::W_3d(Ker::Rkern/3,1));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/4) , Ker::W_3d(Ker::Rkern/4,1));

    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern)/8 , Ker::W_3d(2*Ker::Rkern,2));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/2)/8 , Ker::W_3d(2*Ker::Rkern/2,2));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/3)/8 , Ker::W_3d(2*Ker::Rkern/3,2));
    _AssertEqual(gen_norm3d*Ker::f(Ker::Rkern/4)/8 , Ker::W_3d(2*Ker::Rkern/4,2));

    // test df <-> dW scale relations
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern) , Ker::dW_3d(Ker::Rkern,1));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/2) , Ker::dW_3d(Ker::Rkern/2,1));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/3) , Ker::dW_3d(Ker::Rkern/3,1));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/4) , Ker::dW_3d(Ker::Rkern/4,1));

    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern)/16 , Ker::dW_3d(2*Ker::Rkern,2));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/2)/16 , Ker::dW_3d(2*Ker::Rkern/2,2));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/3)/16 , Ker::dW_3d(2*Ker::Rkern/3,2));
    _AssertEqual(gen_norm3d*Ker::df(Ker::Rkern/4)/16 , Ker::dW_3d(2*Ker::Rkern/4,2));

    // is integral of W == 1 (1d)
    _AssertFloatEqual(1, 
        shammath::integ_riemann_sum<Tscal>(0, Ker::Rkern, dx_int, [](Tscal x) {
            return 2*Ker::W_1d(x,1);
        }
    ),tol)

    // is integral of W == 1 (2d)
    _AssertFloatEqual(1, 
        shammath::integ_riemann_sum<Tscal>(0, Ker::Rkern, dx_int, [](Tscal x) {
            return 2*shambase::Constants<Tscal>::pi*x* Ker::W_2d(x,1);
        }
    ),tol)

    // is integral of W == 1 (3d)
    _AssertFloatEqual(1, 
        shammath::integ_riemann_sum<Tscal>(0, Ker::Rkern, dx_int, [](Tscal x) {
            return 4*shambase::Constants<Tscal>::pi*x*x* Ker::W_3d(x,1);
        }
    ),tol)

    // is df = f' ?
    Tscal L2_sum = 0;
    Tscal step = 0.01;
    for (Tscal x = 0; x < Ker::Rkern; x += step) {
        Tscal diff = Ker::df(x) - 
            shammath::derivative_upwind<Tscal>(x, dx, [](Tscal x) {
                return Ker::f(x);
            });
        diff *= gen_norm3d;
        L2_sum += diff*diff*step;
    }
    _AssertFloatEqual(L2_sum, 0, tol)

}

TestStart(Unittest, "shammath/sphkernels/M4", validateM4kernel, 1){
    validate_kernel_3d<shammath::M4<f32>>(1e-3,1e-4,1e-3);
    validate_kernel_3d<shammath::M4<f64>>(1e-5,1e-5,1e-5);
}

TestStart(Unittest, "shammath/sphkernels/M5", validateM5kernel, 1){
    validate_kernel_3d<shammath::M5<f32>>(1e-3,1e-4,1e-3);
    validate_kernel_3d<shammath::M5<f64>>(1e-5,1e-5,1e-5);
}

TestStart(Unittest, "shammath/sphkernels/M6", validateM6kernel, 1){
    validate_kernel_3d<shammath::M6<f32>>(1e-3,1e-3,1e-3);
    validate_kernel_3d<shammath::M6<f64>>(1e-5,1e-5,1e-5);
}



struct Outplot{
    std::vector<f64> val_W1;
    std::vector<f64> val_dW1;
    std::vector<f64> val_dW1_num;
};

template<class Ker>
Outplot gen_plot(std::vector<f64> & xin){
    Outplot out;

    using Tscal = typename Ker::Tscal;

    for (f64 x : xin) {
        out.val_W1.push_back(Ker::W_3d(x,1));
        out.val_dW1.push_back(Ker::dW_3d(x,1));
        out.val_dW1_num.push_back(
            shammath::derivative_upwind<Tscal>(x, 0.0001, [](Tscal x) {
                return Ker::W_3d(x,1);
            })
        );
    }

    return out;
}

TestStart(Analysis, "shammath/sphkernels_plotall", plotkernels, 1){

    std::vector<f64> X;

    f64 step = 0.01;
    for (f64 x = 0; x < 3; x += step) {
        X.push_back(x);
    }

    Outplot m4 = gen_plot<shammath::M4<f64>>(X);
    Outplot m5 = gen_plot<shammath::M5<f64>>(X);
    Outplot m6 = gen_plot<shammath::M6<f64>>(X);

    PyScriptHandle hdnl{};

    hdnl.data()["X"] = X;

    hdnl.data()["m4"] = m4.val_W1;
    hdnl.data()["m4_d"] = m4.val_dW1;
    hdnl.data()["m4_dnum"] = m4.val_dW1_num;

    hdnl.data()["m5"] = m5.val_W1;
    hdnl.data()["m5_d"] = m5.val_dW1;
    hdnl.data()["m5_dnum"] = m5.val_dW1_num;

    hdnl.data()["m6"] = m6.val_W1;
    hdnl.data()["m6_d"] = m6.val_dW1;
    hdnl.data()["m6_dnum"] = m6.val_dW1_num;


    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,6))

        axs[0,0].plot(X,m4,c = 'black',label = "W")
        axs[0,0].plot(X,m4_d,'--',c = 'black',label = "dW")

        axs[0,1].plot(X,m5,c = 'black',label = "W")
        axs[0,1].plot(X,m5_d,'--',c = 'black',label = "dW")

        axs[0,2].plot(X,m6,c = 'black',label = "W")
        axs[0,2].plot(X,m6_d,'--',c = 'black',label = "dW")

        axs[0,0].set_title('M4')
        axs[0,1].set_title('M5')
        axs[0,2].set_title('M6')
        axs[1,0].set_title('C2')
        axs[1,1].set_title('C4')
        axs[1,2].set_title('C6')

        axs[0,0].set_xlabel(r"$x$")
        axs[0,1].set_xlabel(r"$x$")
        axs[0,2].set_xlabel(r"$x$")
        axs[1,0].set_xlabel(r"$x$")
        axs[1,1].set_xlabel(r"$x$")
        axs[1,2].set_xlabel(r"$x$")

        axs[0,0].legend()

        plt.tight_layout()

        plt.savefig("sph_kernels.pdf")

    )");
    
}





template<class Ker>
f64 benchmark_sph_kernel(u32 N){

    using Tscal = typename Ker::Tscal;

    sycl::buffer<Tscal> dist = shamalgs::random::mock_buffer<Tscal>(0x111,N,0,Ker::Rkern);
    sycl::buffer<Tscal> result (N);

    shamsys::instance::get_compute_queue().wait_and_throw();

    return shambase::timeit([&](){

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor x {dist, cgh, sycl::read_only};
            sycl::accessor f {result, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, N, "test sph kernel", [=](u32 i){
                f[i] = Ker::W_3d(x[i],1);
            });

        }).wait();

    }, 4);

}

TestStart(Benchmark, "shammath/sphkernels_performance", kernelperf, 1){
    f64 m6_f32 = benchmark_sph_kernel<shammath::M6<f32>>(10000000);
    f64 m6_f64 = benchmark_sph_kernel<shammath::M6<f64>>(10000000);
    f64 m4_f32 = benchmark_sph_kernel<shammath::M4<f32>>(10000000);
    f64 m4_f64 = benchmark_sph_kernel<shammath::M4<f64>>(10000000);
    f64 m5_f32 = benchmark_sph_kernel<shammath::M5<f32>>(10000000);
    f64 m5_f64 = benchmark_sph_kernel<shammath::M5<f64>>(10000000);

    
    PyScriptHandle hdnl{};

    hdnl.data()["m6_f32"]   = m6_f32 ;
    hdnl.data()["m6_f64"]   = m6_f64 ;
    hdnl.data()["m5_f32"]   = m5_f32 ;
    hdnl.data()["m5_f64"]   = m5_f64 ;
    hdnl.data()["m4_f32"]   = m4_f32 ;
    hdnl.data()["m4_f64"]   = m4_f64 ;


    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        data_f32 = [m4_f32,m6_f32]
        data_f64 = [m4_f64,m6_f64]
        labels = ['M4','M6']

        plt.xticks(range(len(labels)), labels)
        plt.xlabel('Kernel')
        plt.ylabel('benchmark time')

        width = 0.3
        plt.bar(np.arange(len(data_f32)), data_f32, width=width)
        plt.bar(np.arange(len(data_f64))+width, data_f64, width=width)

        plt.tight_layout()

        plt.savefig("sph_kernel_performance.pdf")

    )");
}