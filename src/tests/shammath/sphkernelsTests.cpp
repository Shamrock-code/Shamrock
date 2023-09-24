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
inline void validate_kernel_3d(typename Ker::Tscal tol,typename Ker::Tscal dx){

    using Tscal = typename Ker::Tscal;

    _AssertEqual(Ker::f(Ker::Rkern) , 0);
    _AssertEqual(Ker::W(Ker::Rkern,1) , 0);

    _AssertFloatEqual(1, 
        shammath::integ_riemann_sum<Tscal>(0, Ker::Rkern, 0.01, [](Tscal x) {
            return 4*shambase::Constants<Tscal>::pi*x*x* Ker::W(x,1);
        }
    ),tol)


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
    validate_kernel_3d<shammath::M4<f32>>(1e-4,1e-4);
    validate_kernel_3d<shammath::M4<f64>>(1e-5,1e-5);
}

TestStart(Unittest, "shammath/sphkernels/M6", validateM6kernel, 1){
    validate_kernel_3d<shammath::M6<f32>>(1e-3,1e-4);
    validate_kernel_3d<shammath::M6<f64>>(1e-5,1e-5);
}