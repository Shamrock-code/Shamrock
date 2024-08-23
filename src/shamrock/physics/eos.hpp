// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file eos.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * \todo move to shammodels generic
 */

#include "shambase/aliases_float.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasFp16.hpp"
#include <memory>

enum EquationsOfStateType { isothermal };

template<class flt>
class EquationsOfState {

    static_assert(
        std::is_same<flt, f16>::value || std::is_same<flt, f32>::value
            || std::is_same<flt, f64>::value,
        "EquationsOfState : floating point type should be one of (f16,f32,f64)");

    inline static flt eos_isothermal(flt cs, flt rho) { return cs * cs * rho; }

    inline static flt apply_eos_isothermal(
        sycl::queue &queue,
        flt cs,
        flt part_mass,
        sycl::buffer<flt> &h_buf,
        sycl::buffer<flt> &out_pressure) {
        sycl::range range_npart{h_buf.size()};

        queue.submit([&](sycl::handler &cgh) {
            auto h = h_buf.get_access<sycl::access::mode::read>(cgh);
            auto p = out_pressure.get_access<sycl::access::mode::discard_write>(cgh);

            auto _cs   = cs;
            auto pmass = part_mass;

            cgh.parallel_for(range_npart, [=](sycl::item<1> item) {
                p[item] = eos_isothermal(_cs, rho_h(pmass, h[item]));
            });
        });
    }
};
