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

    constexpr f64 cs0    = 0.19;
    constexpr f64 rho0   = 1e-13;
    constexpr f64 gamma0 = 5. / 3.;
} // namespace

NEW_TEST(Unittest, "shamphys/eos/barotropic/limits", 1) {

    // well below rho0: isothermal, P = cs0^2 rho and cs = cs0
    f64 rho_lo = 1e-10 * rho0;
    REQUIRE_FLOAT_EQUAL(EOS::pressure(cs0, rho0, gamma0, rho_lo) / (cs0 * cs0 * rho_lo), 1., 1e-6);
    REQUIRE_FLOAT_EQUAL(EOS::soundspeed(cs0, rho0, gamma0, rho_lo) / cs0, 1., 1e-6);

    // well above rho0: polytrope P = K rho^gamma with K = cs0^2 rho0^(1 - gamma),
    // and cs^2 = gamma P / rho
    f64 rho_hi = 1e10 * rho0;
    f64 K      = cs0 * cs0 * std::pow(rho0, 1. - gamma0);
    f64 P_hi   = EOS::pressure(cs0, rho0, gamma0, rho_hi);
    REQUIRE_FLOAT_EQUAL(P_hi / (K * std::pow(rho_hi, gamma0)), 1., 1e-6);

    f64 cs_hi = EOS::soundspeed(cs0, rho0, gamma0, rho_hi);
    REQUIRE_FLOAT_EQUAL(cs_hi * cs_hi / (gamma0 * P_hi / rho_hi), 1., 1e-6);

    // at rho0 the adiabatic and isothermal contributions are equal
    REQUIRE_FLOAT_EQUAL(EOS::pressure(cs0, rho0, gamma0, rho0) / (2 * cs0 * cs0 * rho0), 1., 1e-14);
}

NEW_TEST(Unittest, "shamphys/eos/barotropic/soundspeed_is_dP_drho", 1) {

    // cs^2 must be dP/drho, checked with a centered finite difference across the transition
    for (f64 x : {1e-3, 0.1, 0.5, 1., 2., 10., 1e3}) {
        f64 rho = x * rho0;
        f64 eps = 1e-5 * rho;

        f64 dP_drho = (EOS::pressure(cs0, rho0, gamma0, rho + eps)
                       - EOS::pressure(cs0, rho0, gamma0, rho - eps))
                      / (2 * eps);
        f64 cs      = EOS::soundspeed(cs0, rho0, gamma0, rho);

        REQUIRE_FLOAT_EQUAL(cs * cs / dP_drho, 1., 1e-8);
    }
}

NEW_TEST(Unittest, "shamphys/eos/barotropic/matches_FluidStateBarotropic", 1) {

    // the Ramses solver uses shammath::FluidStateBarotropic, both must stay identical
    shammath::FluidStateBarotropic<f64_3> fspec{rho0, cs0, gamma0};

    for (f64 x : {1e-6, 1e-2, 1., 1e2, 1e6}) {
        f64 rho = x * rho0;

        shammath::PrimState<f64_3> prim{};
        prim.rho   = rho;
        prim.press = fspec.pressure(rho);

        REQUIRE_FLOAT_EQUAL(EOS::pressure(cs0, rho0, gamma0, rho) / fspec.pressure(rho), 1., 1e-14);
        REQUIRE_FLOAT_EQUAL(
            EOS::soundspeed(cs0, rho0, gamma0, rho) / fspec.sound_speed(prim), 1., 1e-14);
    }
}
