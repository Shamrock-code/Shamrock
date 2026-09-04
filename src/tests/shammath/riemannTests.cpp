// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shamtest/shamtest.hpp"

NEW_TEST(Unittest, "shammath/flux_symmetry", 1) {

    using Tcons = shammath::ConsState<f64_3>;
    using Tprim = shammath::PrimState<f64_3>;

    constexpr f64 gamma = 1.6666;

    // Riemann solvers now take primitive states directly (see riemann_hll.hpp,
    // riemann_hllc.hpp, riemann_rusanov.hpp), so the reference states below are converted
    // from conservative once, up front.
    Tcons cons1  = {.rho = 1._f64, .rhoe = 1.2_f64, .rhovel = f64_3{1, 0, 0}};
    Tcons cons2  = {.rho = 1.5_f64, .rhoe = 1._f64, .rhovel = f64_3{2, 0, 0}};
    Tprim state1 = shammath::cons_to_prim(cons1, gamma);
    Tprim state2 = shammath::cons_to_prim(cons2, gamma);

    {
        Tcons f1 = shammath::rusanov_flux_x(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_mx(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_y(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_my(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_z(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_mz(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    auto to_prim = [&](Tcons c) {
        return shammath::cons_to_prim(c, gamma);
    };

    Tprim state_xp = to_prim({.rho = 1.1_f64, .rhoe = 0.8_f64, .rhovel = f64_3{1.1, 0, 0}});
    Tprim state_yp = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_zp = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_i  = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_xm = to_prim({.rho = 0.7_f64, .rhoe = 1.2_f64, .rhovel = f64_3{1.1, 0, 0}});
    Tprim state_ym = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_zm = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    {
        Tcons fx = shammath::rusanov_flux_x(state_i, state_xp, gamma);
        shamlog_debug_ln("Riemann Solver", fx.rho, fx.rhovel, fx.rhoe);
        Tcons fy = shammath::rusanov_flux_y(state_i, state_yp, gamma);
        shamlog_debug_ln("Riemann Solver", fy.rho, fy.rhovel, fy.rhoe);
        Tcons fz = shammath::rusanov_flux_z(state_i, state_zp, gamma);
        shamlog_debug_ln("Riemann Solver", fz.rho, fz.rhovel, fz.rhoe);
        Tcons fmx = shammath::rusanov_flux_mx(state_i, state_xm, gamma);
        shamlog_debug_ln("Riemann Solver", fmx.rho, fmx.rhovel, fmx.rhoe);
        Tcons fmy = shammath::rusanov_flux_my(state_i, state_ym, gamma);
        shamlog_debug_ln("Riemann Solver", fmy.rho, fmy.rhovel, fmy.rhoe);
        Tcons fmz = shammath::rusanov_flux_mz(state_i, state_zm, gamma);
        shamlog_debug_ln("Riemann Solver", fmz.rho, fmz.rhovel, fmz.rhoe);
        Tcons sum = fx + fy + fz + fmx + fmy + fmz;
        shamlog_debug_ln("Riemann Solver", "sum=", sum.rho, sum.rhovel, sum.rhoe);
        REQUIRE(sum.rhovel[1] == 0);
        REQUIRE(sum.rhovel[2] == 0);
    }
}
