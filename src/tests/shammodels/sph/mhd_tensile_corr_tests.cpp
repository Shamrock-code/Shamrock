// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/sph/math/mhd.hpp"
#include "shamtest/shamtest.hpp"

using Tvec  = f64_3;
using Tscal = f64;

namespace {

    Tscal pressure_for_beta(Tscal beta, Tvec B, Tscal mu_0) {
        return beta * sycl::dot(B, B) / (2 * mu_0);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/sph/mhd/tensile_corr_factor", 1) {
    using namespace shamrock::sph::mhd;

    const Tscal mu_0 = 1;
    const Tvec B{1, 0, 0};

    // Borve: Bhat = B/2 regardless of beta
    REQUIRE_FLOAT_EQUAL(tensile_corr_factor_borve<Tscal>(), 0.5, 1e-15);

    // Phantom beta switch, f = 1 (beta < 2), (10 - beta)/8 (2 <= beta < 10), 0 otherwise
    auto f = [&](Tscal beta) {
        return tensile_corr_factor_beta_switch(pressure_for_beta(beta, B, mu_0), B, mu_0);
    };
    REQUIRE_FLOAT_EQUAL(f(1), 1.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(f(2), 1.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(f(6), 0.5, 1e-15);
    REQUIRE_FLOAT_EQUAL(f(10), 0.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(f(12), 0.0, 1e-15);

    // no field -> no correction (and no division by zero)
    REQUIRE_EQUAL(tensile_corr_factor_beta_switch(Tscal(1), Tvec{0, 0, 0}, mu_0), Tscal(0));
}

NEW_TEST(Unittest, "shammodels/sph/mhd/tensile_corr_pair", 1) {
    using namespace shamrock::sph::mhd;

    // Pair a-b with a nonzero symmetric divergence of B, beta_a = 1, beta_b = 12.
    const Tscal mu_0 = 1;
    const Tscal m    = 1;
    const Tvec B_a{1, 0, 0};
    const Tvec B_b{2, 0, 0};
    const Tvec grad_W_ha{1, 0, 0};
    const Tvec grad_W_hb{0.5, 0, 0};
    const Tscal sub_fact = 1; // rho^2 * omega

    const Tscal P_a = pressure_for_beta(1, B_a, mu_0);
    const Tscal P_b = pressure_for_beta(12, B_b, mu_0);

    auto corr_a = [&](Tscal factor) {
        return fdiv_b<Tvec, Tscal>(
            m, B_a, B_b, grad_W_ha, grad_W_hb, sub_fact, sub_fact, mu_0, factor);
    };
    // b sees a with the kernel gradients flipped (nabla_b W_ab = -nabla_a W_ab)
    auto corr_b = [&](Tscal factor) {
        return fdiv_b<Tvec, Tscal>(
            m, B_b, B_a, -grad_W_hb, -grad_W_ha, sub_fact, sub_fact, mu_0, factor);
    };

    // full-strength reference: Bhat = B
    const Tvec full_a = corr_a(1);
    const Tvec full_b = corr_b(1);
    REQUIRE(sycl::length(full_a) > 0);
    REQUIRE(sycl::length(full_b) > 0);

    // phantom_beta_switch: full correction for a (beta = 1), none for b (beta = 12)
    const Tvec sw_a = corr_a(tensile_corr_factor_beta_switch(P_a, B_a, mu_0));
    const Tvec sw_b = corr_b(tensile_corr_factor_beta_switch(P_b, B_b, mu_0));
    REQUIRE_FLOAT_EQUAL(sycl::length(sw_a - full_a), 0.0, 1e-14);
    REQUIRE_EQUAL(sycl::length(sw_b), Tscal(0));

    // default (borve): half-strength correction for both
    const Tvec bo_a = corr_a(tensile_corr_factor_borve<Tscal>());
    const Tvec bo_b = corr_b(tensile_corr_factor_borve<Tscal>());
    REQUIRE_FLOAT_EQUAL(sycl::length(bo_a - 0.5 * full_a), 0.0, 1e-14);
    REQUIRE_FLOAT_EQUAL(sycl::length(bo_b - 0.5 * full_b), 0.0, 1e-14);
}
