// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHKernels.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"

namespace shampylib {

    void init_shamrock_math_sphkernels(py::module &m) {

        py::module sphkernel_module = m.def_submodule("sphkernel", "Shamrock sph kernels math lib");

        sphkernel_module.def("M4_Rkern", []() {
            return shammath::M4<f64>::Rkern;
        });
        sphkernel_module.def("M4_f", &shammath::M4<f64>::f);
        sphkernel_module.def("M4_df", &shammath::M4<f64>::df);
        sphkernel_module.def("M4_W1d", &shammath::M4<f64>::W_1d);
        sphkernel_module.def("M4_W2d", &shammath::M4<f64>::W_2d);
        sphkernel_module.def("M4_W3d", &shammath::M4<f64>::W_3d);
        sphkernel_module.def("M4_dW3d", &shammath::M4<f64>::dW_3d);
        sphkernel_module.def("M4_dhW3d", &shammath::M4<f64>::dhW_3d);
        sphkernel_module.def("M4_f3d_integ_z", &shammath::M4<f64>::f3d_integ_z);

        sphkernel_module.def("M6_Rkern", []() {
            return shammath::M6<f64>::Rkern;
        });
        sphkernel_module.def("M6_f", &shammath::M6<f64>::f);
        sphkernel_module.def("M6_df", &shammath::M6<f64>::df);
        sphkernel_module.def("M6_W1d", &shammath::M6<f64>::W_1d);
        sphkernel_module.def("M6_W2d", &shammath::M6<f64>::W_2d);
        sphkernel_module.def("M6_W3d", &shammath::M6<f64>::W_3d);
        sphkernel_module.def("M6_dW3d", &shammath::M6<f64>::dW_3d);
        sphkernel_module.def("M6_dhW3d", &shammath::M6<f64>::dhW_3d);
        sphkernel_module.def("M6_f3d_integ_z", &shammath::M6<f64>::f3d_integ_z);

        sphkernel_module.def("M8_Rkern", []() {
            return shammath::M8<f64>::Rkern;
        });
        sphkernel_module.def("M8_f", &shammath::M8<f64>::f);
        sphkernel_module.def("M8_df", &shammath::M8<f64>::df);
        sphkernel_module.def("M8_W1d", &shammath::M8<f64>::W_1d);
        sphkernel_module.def("M8_W2d", &shammath::M8<f64>::W_2d);
        sphkernel_module.def("M8_W3d", &shammath::M8<f64>::W_3d);
        sphkernel_module.def("M8_dW3d", &shammath::M8<f64>::dW_3d);
        sphkernel_module.def("M8_dhW3d", &shammath::M8<f64>::dhW_3d);
        sphkernel_module.def("M8_f3d_integ_z", &shammath::M8<f64>::f3d_integ_z);
    }

} // namespace shampylib
