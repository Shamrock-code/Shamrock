// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file density.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
namespace shamrock::sph {

    template<class flt>
    inline flt rho_h(flt m, flt h, flt hfact) {
        return m * (hfact / h) * (hfact / h) * (hfact / h);
    }

    template<class flt, i32 dim = 3>
    inline flt h_rho(flt m, flt rho, flt hfact) {
        return hfact / sycl::rootn(rho / m, dim);
    }

    template<class flt, i32 dim = 3>
    inline flt newtown_iterate_new_h(flt rho_ha, flt rho_sum, flt sumdWdh, flt h_a) {
        flt f_iter  = rho_sum - rho_ha;
        flt df_iter = sumdWdh + dim * rho_ha / h_a;

        // flt omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
        // flt new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

        return h_a - f_iter / df_iter;
    }

    template<class flt>
    inline flt num_dens_h(flt h, flt hfact) {
        return (hfact / h) * (hfact / h) * (hfact / h);
    }

    template<class flt, i32 dim = 3>
    inline flt h_num_dens(flt rho, flt hfact) {
        return hfact / sycl::rootn(rho, dim);
    }

    template<class flt, i32 dim = 3>
    inline flt newtown_iterate_new_h_num_dens(
        flt num_dens_ha, flt num_dens_sum, flt sumdWdh, flt h_a) {
        flt f_iter  = num_dens_sum - num_dens_ha;
        flt df_iter = sumdWdh + dim * num_dens_ha / h_a;

        // flt omega_a = 1 + (h_a/(3*rho_ha))*sumdWdh;
        // flt new_h = h_a - (rho_ha - rho_sum)/((-3*rho_ha/h_a)*omega_a);

        return h_a - f_iter / df_iter;
    }
} // namespace shamrock::sph
