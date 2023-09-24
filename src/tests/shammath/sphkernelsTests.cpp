// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shambase/Constants.hpp"
#include "shammath/derivatives.hpp"
#include "shammath/integrator.hpp"
#include "shamtest/shamtest.hpp"

#include "shammath/sphkernels.hpp"



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
        L2_sum += diff*diff*step;
    }
    _AssertFloatEqual(L2_sum, 0, tol)



}

TestStart(Unittest, "shammath/sphkernels/M4", validateM4kernel, 1){
    validate_kernel_3d<shammath::M4<f32>>(1e-3,1e-4,1e-3);
    validate_kernel_3d<shammath::M4<f64>>(1e-5,1e-5,1e-5);
}

TestStart(Unittest, "shammath/sphkernels/M6", validateM6kernel, 1){
    validate_kernel_3d<shammath::M6<f32>>(1e-3,1e-4,1e-3);
    validate_kernel_3d<shammath::M6<f64>>(1e-5,1e-5,1e-5);
}