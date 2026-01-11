// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SodTube.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sod tube analytical solution adapted from a script of Leodasce Sewanou
 */

#include "shamphys/SodTube.hpp"
#include "shammath/derivatives.hpp"
#include "shammath/solve.hpp"
#include <cmath>
#include <iostream>

f64 shamphys::SodTube::soundspeed(f64 P, f64 rho) { return std::sqrt(gamma * P / rho); };

shamphys::SodTube::SodTube(f64 _gamma, f64 _rho_1, f64 _P_1, f64 _rho_5, f64 _P_5)
    : gamma(_gamma), rho_1(_rho_1), P_1(_P_1), rho_5(_rho_5), P_5(_P_5) {
    c_1 = soundspeed(P_1, rho_1);
    c_5 = soundspeed(P_5, rho_5);

    if (P_5 > P_1) {
        throw "not correct";
    }
}

f64 shamphys::SodTube::solve_P_4() {

    auto shock_tube_function = [=](f64 P_4) {
        f64 z = (P_4 / P_5 - 1.);

        f64 gm1 = gamma - 1.0;
        f64 gp1 = gamma + 1.0;
        f64 g2  = 2.0 * gamma;

        f64 fact1 = gm1 / g2 * (c_5 / c_1) * z / std::sqrt(1. + gp1 / g2 * z);
        f64 fact  = std::pow(1.0 - fact1, g2 / gm1);

        return P_1 * fact - P_4;
    };

    auto df = [=](f64 P_4) {
        return shammath::derivative_upwind<f64>(P_4, 1e-6, shock_tube_function);
    };

    return shammath::newton_rhaphson<f64>(shock_tube_function, df, 1e-6, P_1);
}

auto shamphys::SodTube::get_value(f64 t, f64 x) -> field_val {

    f64 P_4 = solve_P_4();

    f64 z = (P_4 / P_5 - 1.);

    f64 gm1     = gamma - 1.0;
    f64 gp1     = gamma + 1.0;
    f64 gmfact1 = 0.5 * gm1 / gamma;
    f64 gmfact2 = 0.5 * gp1 / gamma;

    f64 fact = std::sqrt(1. + gmfact2 * z);

    f64 vx_4  = c_5 * z / (gamma * fact);
    f64 rho_4 = rho_5 * (1. + gmfact2 * z) / (1. + gmfact1 * z);

    // shock speed

    f64 w = c_5 * fact;

    // compute values at foot of rarefaction
    f64 P_3   = P_4;
    f64 vx_3  = vx_4;
    f64 rho_3 = rho_1 * std::pow(P_3 / P_1, 1. / gamma);

    // compute positions
    f64 c3 = soundspeed(P_3, rho_3);

    f64 xhd, xft, xcd, xsh;

    f64 xi = 0.;
    xsh    = xi + w * t;
    xcd    = xi + vx_3 * t;
    xft    = xi + (vx_3 - c3) * t;
    xhd    = xi - c_1 * t;

    /*
        1 : Head of rarefaction
        2 : RAREFACTION
        3 : Foot of rarefaction,
        4 : Contact discontinuity,
        5 : Shock,
    */

    f64 rho, p, vx;

    if (x < xhd) {
        rho = rho_1;
        p   = P_1;
        vx  = vx_1;
    } else if (x < xft) {
        vx          = 2. / gp1 * (c_1 + (x - xi) / t);
        f64 locfact = 1. - 0.5 * gm1 * vx / c_1;
        rho         = rho_1 * std::pow(locfact, 2. / gm1);
        p           = P_1 * std::pow(locfact, 2. * gamma / gm1);
    } else if (x < xcd) {
        rho = rho_3;
        p   = P_3;
        vx  = vx_3;

    } else if (x < xsh) {
        rho = rho_4;
        p   = P_4;
        vx  = vx_4;

    } else {
        rho = rho_5;
        p   = P_5;
        vx  = vx_5;
    }

    field_val ret;
    ret.P   = p;
    ret.vx  = vx;
    ret.rho = rho;
    return ret;

    return {};
}
