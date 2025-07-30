// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/flux_symmetry", flux_rotate, 1) {

    using Tcons = shammath::ConsState<f64_3>;

    Tcons state1 = {1._f64, 1.2_f64, f64_3{1, 0, 0}};
    Tcons state2 = {1.5_f64, 1._f64, f64_3{2, 0, 0}};

    {
        Tcons f1 = shammath::rusanov_flux_x(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_mx(state2, state1, 1.6666);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_y(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_my(state2, state1, 1.6666);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_z(state1, state2, 1.6666);
        Tcons f2 = shammath::rusanov_flux_mz(state2, state1, 1.6666);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    Tcons state_xp = {1.1_f64, 0.8_f64, f64_3{1.1, 0, 0}};
    Tcons state_yp = {1._f64, 1._f64, f64_3{1, 0, 0}};
    Tcons state_zp = {1._f64, 1._f64, f64_3{1, 0, 0}};
    Tcons state_i  = {1._f64, 1._f64, f64_3{1, 0, 0}};
    Tcons state_xm = {0.7_f64, 1.2_f64, f64_3{1.1, 0, 0}};
    Tcons state_ym = {1._f64, 1._f64, f64_3{1, 0, 0}};
    Tcons state_zm = {1._f64, 1._f64, f64_3{1, 0, 0}};
    {
        Tcons fx = shammath::rusanov_flux_x(state_i, state_xp, 1.6666);
        shamlog_debug_ln("Riemman Solver", fx.rho, fx.rhovel, fx.rhoe);
        Tcons fy = shammath::rusanov_flux_y(state_i, state_yp, 1.6666);
        shamlog_debug_ln("Riemman Solver", fy.rho, fy.rhovel, fy.rhoe);
        Tcons fz = shammath::rusanov_flux_z(state_i, state_zp, 1.6666);
        shamlog_debug_ln("Riemman Solver", fz.rho, fz.rhovel, fz.rhoe);
        Tcons fmx = shammath::rusanov_flux_mx(state_i, state_xm, 1.6666);
        shamlog_debug_ln("Riemman Solver", fmx.rho, fmx.rhovel, fmx.rhoe);
        Tcons fmy = shammath::rusanov_flux_my(state_i, state_ym, 1.6666);
        shamlog_debug_ln("Riemman Solver", fmy.rho, fmy.rhovel, fmy.rhoe);
        Tcons fmz = shammath::rusanov_flux_mz(state_i, state_zm, 1.6666);
        shamlog_debug_ln("Riemman Solver", fmz.rho, fmz.rhovel, fmz.rhoe);
        Tcons sum = fx + fy + fz + fmx + fmy + fmz;
        shamlog_debug_ln("Riemman Solver", "sum=", sum.rho, sum.rhovel, sum.rhoe);
        REQUIRE(sum.rhovel[1] == 0);
        REQUIRE(sum.rhovel[2] == 0);
    }
}
