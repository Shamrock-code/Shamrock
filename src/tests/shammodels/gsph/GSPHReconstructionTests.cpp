// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHReconstructionTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Unit tests for the Inutsuka (2002) effective face interpolation (V2_ij, s*)
 *
 * Tests cover:
 * - Symmetric pair (equal volume/h) gives V2 = D^2 and s* = 0
 * - General case matches the closed-form V2/s* expressions directly
 * - The Inutsuka force contribution satisfies Newton's 3rd law
 */

#include "shammodels/gsph/math/forces.hpp"
#include "shammodels/gsph/math/reconstruction.hpp"
#include "shamtest/shamtest.hpp"

namespace {

    using namespace shammodels::gsph;

    //==========================================================================
    // SCENARIO: symmetric pair (equal volume and smoothing length)
    //==========================================================================

    void test_lin_v2_sast_symmetric_pair() {
        using Tscal = f64;

        const Tscal vol     = 1.5;
        const Tscal h       = 0.2;
        const Tscal rab_inv = 1.0 / 0.1;

        auto face = lin_v2_sast_ij<Tscal>(vol, vol, h, h, rab_inv);

        // C = 0 when vol_a == vol_b, so V2 reduces to D^2 and s* = 0
        REQUIRE_FLOAT_EQUAL_NAMED("V2 == D^2 for symmetric pair", face.V2, vol * vol, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("s* == 0 for symmetric pair", face.s_ast, 0.0, 1e-12);
    }

    //==========================================================================
    // SCENARIO: general pair matches the closed-form expression
    //==========================================================================

    void test_lin_v2_sast_general_pair() {
        using Tscal = f64;

        const Tscal vol_a   = 1.2;
        const Tscal vol_b   = 0.8;
        const Tscal h_a     = 0.3;
        const Tscal h_b     = 0.25;
        const Tscal rab     = 0.4;
        const Tscal rab_inv = 1.0 / rab;

        auto face = lin_v2_sast_ij<Tscal>(vol_a, vol_b, h_a, h_b, rab_inv);

        const Tscal C_expect    = (vol_a - vol_b) / rab;
        const Tscal D_expect    = 0.5 * (vol_a + vol_b);
        const Tscal h2_expect   = 0.5 * (h_a * h_a + h_b * h_b);
        const Tscal V2_expect   = 0.25 * h2_expect * C_expect * C_expect + D_expect * D_expect;
        const Tscal sast_expect = 0.5 * h2_expect * C_expect * D_expect / V2_expect;

        REQUIRE_FLOAT_EQUAL_NAMED("V2 matches closed form", face.V2, V2_expect, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("s* matches closed form", face.s_ast, sast_expect, 1e-12);
    }

    //==========================================================================
    // SCENARIO: Inutsuka force contribution satisfies Newton's 3rd law
    //==========================================================================

    void test_inutsuka_force_newtons_third_law() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal p_star = 1.3;
        const Tscal v_star = 0.2;
        const Tscal V2_ij  = 0.7;
        const Tvec v_a     = Tvec{0, 0, 0};

        // Force on a from b (b is at +x from a)
        Tvec grad_W_ab = Tvec{-1.5, 0, 0};
        Tvec r_ab_unit = Tvec{1, 0, 0};
        Tvec dv_a      = Tvec{0, 0, 0};
        Tscal du_a     = 0;

        add_gsph_force_contribution_inutsuka<Tvec, Tscal>(
            m, p_star, v_star, V2_ij, grad_W_ab, r_ab_unit, v_a, dv_a, du_a);

        // Force on b from a: pair axis reverses, so grad_W and r_ab_unit flip sign
        Tvec grad_W_ba = -grad_W_ab;
        Tvec r_ba_unit = -r_ab_unit;
        Tvec dv_b      = Tvec{0, 0, 0};
        Tscal du_b     = 0;

        add_gsph_force_contribution_inutsuka<Tvec, Tscal>(
            m, p_star, v_star, V2_ij, grad_W_ba, r_ba_unit, v_a, dv_b, du_b);

        Tvec total = dv_a + dv_b;
        REQUIRE_FLOAT_EQUAL_NAMED("x momentum conserved", total[0], 0.0, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("y momentum conserved", total[1], 0.0, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("z momentum conserved", total[2], 0.0, 1e-12);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/gsph/reconstruction/lin_v2_sast_symmetric", 1) {
    test_lin_v2_sast_symmetric_pair();
}

NEW_TEST(Unittest, "shammodels/gsph/reconstruction/lin_v2_sast_general", 1) {
    test_lin_v2_sast_general_pair();
}

NEW_TEST(Unittest, "shammodels/gsph/reconstruction/inutsuka_force_newton3", 1) {
    test_inutsuka_force_newtons_third_law();
}
