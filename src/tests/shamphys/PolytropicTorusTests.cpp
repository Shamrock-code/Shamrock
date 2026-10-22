// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamphys/PolytropicTorus.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <stdexcept>

NEW_TEST(Unittest, "shamphys/PolytropicTorus::papaloizou_pringle", 1) {
    // constant angular momentum torus around a point mass, GM = r0 = 1:
    // Phi_eff = -1/r + 1/(2 varpi^2), the edges are the roots of
    // phi_eff_0 r_cyl^2 + r_cyl - 1/2 = 0
    f64 r_in = 0.7;
    shamphys::PolytropicTorus torus(1.0, 0.0, 1.0, r_in, 0.0, 1.5);

    f64 phi_eff_0 = -1 / r_in + 1 / (2 * r_in * r_in);

    REQUIRE_FLOAT_EQUAL(torus.get_r_cusp(), 0.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(torus.get_r_out(), -1 / (2 * phi_eff_0 * r_in), 1e-10);
    REQUIRE_FLOAT_EQUAL(torus.get_W0(), phi_eff_0 + 0.5, 1e-14);
    REQUIRE_FLOAT_EQUAL(torus.rho_shape(1.0, 0.0), 1.0, 1e-14);
    REQUIRE_FLOAT_EQUAL(torus.v_phi(1.0), 1.0, 1e-14);
    REQUIRE_FLOAT_EQUAL(torus.v_phi(2.0), 0.5, 1e-14);

    // surface above the pressure maximum: -1/sqrt(1 + z^2) + 1/2 = phi_eff_0
    f64 z_r0 = std::sqrt(1 / ((0.5 - phi_eff_0) * (0.5 - phi_eff_0)) - 1);
    REQUIRE(torus.enthalpy(1.0, z_r0 * (1 - 1e-6)) > 0);
    REQUIRE(torus.enthalpy(1.0, z_r0 * (1 + 1e-6)) == 0);
    REQUIRE(torus.get_z_max() >= z_r0 * (1 - 1e-9));

    REQUIRE(torus.volume_integral() > 0);
    REQUIRE(torus.pressure_integral() > 0);
}

NEW_TEST(Unittest, "shamphys/PolytropicTorus::paczynski_wiita", 1) {
    // Kato et al. (2004) like torus in units of r_s (GM = 1/2, r_s = 1)
    f64 GM = 0.5, r_s = 1.0, r0 = 40.0;
    shamphys::PolytropicTorus torus(GM, r_s, r0, 20.0, 0.46, 3.0);

    // Keplerian at the pressure maximum, which is an extremum of Phi_eff
    REQUIRE_FLOAT_EQUAL(torus.v_phi(r0), std::sqrt(GM * r0) / (r0 - r_s), 1e-14);
    REQUIRE_FLOAT_EQUAL(torus.grad_phi_eff(r0, 0).first, 0.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(torus.rho_shape(r0, 0), 1.0, 1e-14);

    // the cusp is the other extremum of Phi_eff on the midplane, inside r_in
    f64 r_cusp = torus.get_r_cusp();
    REQUIRE(r_cusp > r_s && r_cusp < 20.0);
    REQUIRE_FLOAT_EQUAL(torus.grad_phi_eff(r_cusp, 0).first, 0.0, 1e-9);

    // edges and extent
    REQUIRE_FLOAT_EQUAL(torus.enthalpy(20.0, 0), 0.0, 1e-15);
    REQUIRE_FLOAT_EQUAL(torus.phi_eff(torus.get_r_out(), 0), torus.phi_eff(20.0, 0), 1e-15);
    REQUIRE(torus.rho_shape(20.0 * (1 - 1e-6), 0) == 0);
    REQUIRE(torus.rho_shape(torus.get_r_out() * (1 + 1e-6), 0) == 0);
    REQUIRE(torus.rho_shape(r0, torus.get_z_max() * (1 + 1e-6)) == 0);

    // gradient of rho_shape against centred finite differences (limited by the cancellation
    // in W = phi_eff_0 - Phi_eff, W0 / phi_eff_0 ~ 0.1 here)
    f64 e = 1e-6;
    for (auto [r_cyl, z] : {std::pair<f64, f64>{30, 1}, {45, 2}, {55, -1.5}}) {
        auto [dr_cyl, dz] = torus.grad_rho_shape(r_cyl, z);
        f64 dr_cyl_fd = (torus.rho_shape(r_cyl + e, z) - torus.rho_shape(r_cyl - e, z)) / (2 * e);
        f64 dz_fd     = (torus.rho_shape(r_cyl, z + e) - torus.rho_shape(r_cyl, z - e)) / (2 * e);
        REQUIRE_FLOAT_EQUAL(dr_cyl, dr_cyl_fd, 1e-6);
        REQUIRE_FLOAT_EQUAL(dz, dz_fd, 1e-6);
    }

    // the poloidal field derives from a vector potential: div B = 0
    f64 A_cut = 0.2;
    e         = 1e-5;
    for (auto [r_cyl, z] : {std::pair<f64, f64>{30, 1}, {45, 2}, {40, -0.5}}) {
        auto B = [&](f64 r_cyl, f64 z) {
            return torus.b_poloidal_unit(r_cyl, z, A_cut);
        };
        f64 div_B  = ((r_cyl + e) * B(r_cyl + e, z).first - (r_cyl - e) * B(r_cyl - e, z).first)
                         / (2 * e * r_cyl)
                     + (B(r_cyl, z + e).second - B(r_cyl, z - e).second) / (2 * e);
        f64 B_norm = std::hypot(B(r_cyl, z).first, B(r_cyl, z).second);
        REQUIRE(B_norm > 0);
        REQUIRE_FLOAT_EQUAL(div_B / B_norm, 0.0, 1e-6);
    }
    REQUIRE(torus.b_poloidal_unit_sq_integral(A_cut) > 0);
    REQUIRE_FLOAT_EQUAL(torus.b_poloidal_unit_sq_integral(1.0), 0.0, 1e-30);
}

NEW_TEST(Unittest, "shamphys/PolytropicTorus::invalid", 1) {
    // inner edge inside the cusp (r_cusp ~ 17.9 r_s)
    REQUIRE_EXCEPTION_THROW(
        shamphys::PolytropicTorus(0.5, 1.0, 40.0, 10.0, 0.46, 3.0), std::invalid_argument);
    // inner edge beyond the pressure maximum
    REQUIRE_EXCEPTION_THROW(
        shamphys::PolytropicTorus(0.5, 1.0, 40.0, 50.0, 0.46, 3.0), std::invalid_argument);
    // Keplerian angular momentum law
    REQUIRE_EXCEPTION_THROW(
        shamphys::PolytropicTorus(0.5, 1.0, 40.0, 20.0, 0.5, 3.0), std::invalid_argument);
    // pressure maximum inside the marginally stable orbit (3 r_s for a = 0)
    REQUIRE_EXCEPTION_THROW(
        shamphys::PolytropicTorus(0.5, 1.0, 2.5, 2.2, 0.0, 3.0), std::invalid_argument);
}
