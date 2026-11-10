// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammath/riemann_common.hpp"
#include "shamphys/eos.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>

namespace {
    using EOS = shamphys::EOS_Barotropic<f64>;

    constexpr f64 cs0      = 0.19;
    constexpr f64 rho_crit = 1e-13;
    constexpr f64 gamma_ad = 5. / 3.;
} // namespace

NEW_TEST(Unittest, "shamphys/eos/barotropic/limits", 1) {

    // well below rho_crit: isothermal, P = cs0^2 rho and cs = cs0
    f64 rho_lo = 1e-10 * rho_crit;
    REQUIRE_FLOAT_EQUAL(
        EOS::pressure(cs0, rho_crit, gamma_ad, rho_lo) / (cs0 * cs0 * rho_lo), 1., 1e-6);
    REQUIRE_FLOAT_EQUAL(EOS::soundspeed(cs0, rho_crit, gamma_ad, rho_lo) / cs0, 1., 1e-6);

    // well above rho_crit: polytrope P = K rho^gamma with K = cs0^2 rho_crit^(1 - gamma),
    // and cs^2 = gamma P / rho
    f64 rho_hi = 1e10 * rho_crit;
    f64 K      = cs0 * cs0 * std::pow(rho_crit, 1. - gamma_ad);
    f64 P_hi   = EOS::pressure(cs0, rho_crit, gamma_ad, rho_hi);
    REQUIRE_FLOAT_EQUAL(P_hi / (K * std::pow(rho_hi, gamma_ad)), 1., 1e-6);

    f64 cs_hi = EOS::soundspeed(cs0, rho_crit, gamma_ad, rho_hi);
    REQUIRE_FLOAT_EQUAL(cs_hi * cs_hi / (gamma_ad * P_hi / rho_hi), 1., 1e-6);

    // at rho_crit the adiabatic and isothermal contributions are equal
    REQUIRE_FLOAT_EQUAL(
        EOS::pressure(cs0, rho_crit, gamma_ad, rho_crit) / (2 * cs0 * cs0 * rho_crit), 1., 1e-14);
}

NEW_TEST(Unittest, "shamphys/eos/barotropic/soundspeed_is_dP_drho", 1) {

    // cs^2 must be dP/drho, checked with a centered finite difference across the transition
    for (f64 x : {1e-3, 0.1, 0.5, 1., 2., 10., 1e3}) {
        f64 rho = x * rho_crit;
        f64 eps = 1e-5 * rho;

        f64 dP_drho = (EOS::pressure(cs0, rho_crit, gamma_ad, rho + eps)
                       - EOS::pressure(cs0, rho_crit, gamma_ad, rho - eps))
                      / (2 * eps);
        f64 cs      = EOS::soundspeed(cs0, rho_crit, gamma_ad, rho);

        REQUIRE_FLOAT_EQUAL(cs * cs / dP_drho, 1., 1e-8);
    }
}

NEW_TEST(Unittest, "shamphys/eos/barotropic/matches_FluidStateBarotropic", 1) {

    // the Ramses solver uses shammath::FluidStateBarotropic, both must stay identical
    shammath::FluidStateBarotropic<f64_3> fspec{rho_crit, cs0, gamma_ad};

    for (f64 x : {1e-6, 1e-2, 1., 1e2, 1e6}) {
        f64 rho = x * rho_crit;

        shammath::PrimState<f64_3> prim{};
        prim.rho   = rho;
        prim.press = fspec.pressure(rho);

        REQUIRE_FLOAT_EQUAL(
            EOS::pressure(cs0, rho_crit, gamma_ad, rho) / fspec.pressure(rho), 1., 1e-14);
        REQUIRE_FLOAT_EQUAL(
            EOS::soundspeed(cs0, rho_crit, gamma_ad, rho) / fspec.sound_speed(prim), 1., 1e-14);
    }
}
