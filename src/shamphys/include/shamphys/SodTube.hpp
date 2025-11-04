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
 * @file SodTube.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
namespace shamphys {

    class SodTube {
        public:
        f64 gamma;
        f64 rho_1, rho_5;
        f64 P_1, P_5;
        f64 vx_1, vx_5;

        SodTube(f64 _gamma, f64 _rho_1, f64 _P_1, f64 _rho_5, f64 _P_5);

        struct field_val {
            f64 rho, vx, P;
        };

        field_val get_value(f64 t, f64 x);

        private:
        f64 c_1, c_5;

        f64 soundspeed(f64 P, f64 rho);

        f64 solve_P_4();
    };

} // namespace shamphys
