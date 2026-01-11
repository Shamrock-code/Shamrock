// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHKernels.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"

namespace {
    // this is a container to store the string for the lifetime of the code
    std::vector<std::string> names = {};

    template<typename Kernel>
    void bind_sph_kernel(py::module &m, const std::string &name) {

        // yeah this looks weird, the idea is to first store the string for the lifetime of the code
        // and return its c_str pointer, which is then garanteed to be valid until shutdown

        auto get_name = [&](std::string suffix) -> const char * {
            return names.emplace_back(name + suffix).c_str();
        };

        m.def(get_name("_norm_1d"), []() {
            return Kernel::Generator::norm_1d;
        });
        m.def(get_name("_norm_2d"), []() {
            return Kernel::Generator::norm_2d;
        });
        m.def(get_name("_norm_3d"), []() {
            return Kernel::Generator::norm_3d;
        });
        m.def(get_name("_Rkern"), []() {
            return Kernel::Rkern;
        });
        m.def(get_name("_f"), &Kernel::f);
        m.def(get_name("_df"), &Kernel::df);
        m.def(get_name("_ddf"), &Kernel::ddf);
        m.def(get_name("_phi_tilde_3d"), &Kernel::phi_tilde_3d);
        m.def(get_name("_phi_tilde_3d_prime"), &Kernel::phi_tilde_3d_prime);
        m.def(get_name("_W1d"), &Kernel::W_1d);
        m.def(get_name("_W2d"), &Kernel::W_2d);
        m.def(get_name("_W3d"), &Kernel::W_3d);
        m.def(get_name("_dW3d"), &Kernel::dW_3d);
        m.def(get_name("_ddW3d"), &Kernel::ddW_3d);
        m.def(get_name("_dhW3d"), &Kernel::dhW_3d);
        m.def(get_name("_f3d_integ_z"), &Kernel::f3d_integ_z);
    }

} // namespace

namespace shampylib {

    void init_shamrock_math_sphkernels(py::module &m) {

        py::module sphkernel_module = m.def_submodule("sphkernel", "Shamrock sph kernels math lib");

        bind_sph_kernel<shammath::M4<f64>>(sphkernel_module, "M4");
        bind_sph_kernel<shammath::M5<f64>>(sphkernel_module, "M5");
        bind_sph_kernel<shammath::M6<f64>>(sphkernel_module, "M6");
        bind_sph_kernel<shammath::M7<f64>>(sphkernel_module, "M7");
        bind_sph_kernel<shammath::M8<f64>>(sphkernel_module, "M8");
        bind_sph_kernel<shammath::M9<f64>>(sphkernel_module, "M9");
        bind_sph_kernel<shammath::M10<f64>>(sphkernel_module, "M10");
        bind_sph_kernel<shammath::C2<f64>>(sphkernel_module, "C2");
        bind_sph_kernel<shammath::C4<f64>>(sphkernel_module, "C4");
        bind_sph_kernel<shammath::C6<f64>>(sphkernel_module, "C6");
        bind_sph_kernel<shammath::M4DH<f64>>(sphkernel_module, "M4DH");
        bind_sph_kernel<shammath::M4DH3<f64>>(sphkernel_module, "M4DH3");
        bind_sph_kernel<shammath::M4DH5<f64>>(sphkernel_module, "M4DH5");
        bind_sph_kernel<shammath::M4DH7<f64>>(sphkernel_module, "M4DH7");
        bind_sph_kernel<shammath::M4Shift2<f64>>(sphkernel_module, "M4Shift2");
        bind_sph_kernel<shammath::M4Shift4<f64>>(sphkernel_module, "M4Shift4");
        bind_sph_kernel<shammath::M4Shift8<f64>>(sphkernel_module, "M4Shift8");
        bind_sph_kernel<shammath::M4Shift16<f64>>(sphkernel_module, "M4Shift16");
        bind_sph_kernel<shammath::TGauss3<f64>>(sphkernel_module, "TGauss3");
        bind_sph_kernel<shammath::TGauss5<f64>>(sphkernel_module, "TGauss5");
    }

} // namespace shampylib
