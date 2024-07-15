// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SodTube.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
        f64 gamma_corec;

        SodTube(f64 _gamma, f64 _rho_1, f64 _P_1, f64 _rho_5, f64 _P_5, f64 _gamma_corec = 1);

        struct field_val {
            f64 rho, vx, P;
        };

        field_val get_value(f64 t, f64 x);

        private:
        f64 c_1, c_5;

        f64 soundspeed(f64 P, f64 rho);

        f64 solve_P_4();
    };

    class SodTubeDustTVI {
        SodTube solver;
        f64 epsilon;

        public:
        SodTubeDustTVI(f64 _gamma, f64 _epsilon, f64 _rho_1, f64 _P_1, f64 _rho_5, f64 _P_5)
            : epsilon(_epsilon), solver(_gamma, _rho_1, _P_1, _rho_5, _P_5, (1 - _epsilon)) {}

        struct field_val {
            f64 rho, epsilon, vx, P;
        };

        field_val get_value(f64 t, f64 x) {
            auto res = solver.get_value(t, x);

            field_val ret;
            ret.P       = res.P;
            ret.vx      = res.vx;
            ret.rho     = res.rho;
            ret.epsilon = epsilon;
            return ret;
        }
    };

} // namespace shamphys
