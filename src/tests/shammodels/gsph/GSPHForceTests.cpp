// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHForceTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Comprehensive BDD-style unit tests for GSPH force and energy calculations
 *
 * Tests cover:
 * - Newton's 3rd law (momentum conservation)
 * - Force scaling with p_star
 * - Energy rate calculation (PdV work)
 * - Zero density/omega handling (boundary particles)
 * - Complete GSPH pair interaction
 * - Direction-dependent force behavior
 * - Symmetry properties
 */

#include "shammodels/gsph/math/forces.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <vector>

namespace {

    using namespace shammodels::gsph;

    //==========================================================================
    // SCENARIO: SPH pressure force satisfies Newton's 3rd law
    //==========================================================================

    void test_sph_force_newtons_third_law() {
        using Tvec  = f64_3;
        using Tscal = f64;

        // Test with various configurations
        const std::vector<std::tuple<Tscal, Tscal, Tscal, Tscal>> configs = {
            {1.0, 1.0, 1.0, 1.0},  // Equal particles
            {1.0, 0.5, 1.0, 0.5},  // Different density/pressure
            {2.0, 0.25, 0.5, 2.0}, // Asymmetric
            {1.0, 1.0, 0.1, 0.1},  // Low pressure right
        };

        for (const auto &[rho_a, P_a, rho_b, P_b] : configs) {
            const Tscal m       = 1.0;
            const Tscal omega_a = 1.0;
            const Tscal omega_b = 1.0;

            // Kernel gradient pointing from a to b
            const Tscal Fab = 10.0;
            Tvec nabla_W_ab = Tvec{Fab, 0, 0};

            // Force on a from b
            Tvec F_on_a = shamrock::sph::sph_pressure_symetric<Tvec, Tscal>(
                m,
                rho_a * rho_a,
                rho_b * rho_b,
                P_a,
                P_b,
                omega_a,
                omega_b,
                nabla_W_ab,
                nabla_W_ab);

            // Force on b from a (reversed direction)
            Tvec nabla_W_ba = -nabla_W_ab;
            Tvec F_on_b     = shamrock::sph::sph_pressure_symetric<Tvec, Tscal>(
                m,
                rho_b * rho_b,
                rho_a * rho_a,
                P_b,
                P_a,
                omega_b,
                omega_a,
                nabla_W_ba,
                nabla_W_ba);

            // Newton's 3rd law: F_on_a + F_on_b = 0
            Tvec total = F_on_a + F_on_b;
            REQUIRE_FLOAT_EQUAL_NAMED("x momentum conserved", total[0], 0.0, 1e-12);
            REQUIRE_FLOAT_EQUAL_NAMED("y momentum conserved", total[1], 0.0, 1e-12);
            REQUIRE_FLOAT_EQUAL_NAMED("z momentum conserved", total[2], 0.0, 1e-12);
        }
    }

    //==========================================================================
    // SCENARIO: GSPH force with same kernel satisfies symmetry
    //==========================================================================

    void test_gsph_force_symmetry() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m     = 1.0;
        const Tscal rho   = 1.0;
        const Tscal omega = 1.0;
        const Tscal Fab   = 10.0;

        // Test with different p_star and v_star values
        const std::vector<std::tuple<Tscal, Tscal>> pv_values = {
            {0.5, 0.0},
            {1.0, 0.5},
            {2.0, -0.3},
            {0.1, 1.0},
        };

        for (const auto &[p_star, v_star] : pv_values) {
            // Compute force on a (b is at +x from a)
            Tvec r_ab_unit = Tvec{1, 0, 0};
            Tvec v_a       = Tvec{0, 0, 0};
            Tvec dv_a      = Tvec{0, 0, 0};
            Tscal du_a     = 0;

            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, omega, omega, Fab, Fab, r_ab_unit, v_a, dv_a, du_a);

            // Compute force on b (a is at -x from b)
            Tvec r_ba_unit = -r_ab_unit;
            Tvec v_b       = Tvec{0, 0, 0};
            Tvec dv_b      = Tvec{0, 0, 0};
            Tscal du_b     = 0;

            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, omega, omega, Fab, Fab, r_ba_unit, v_b, dv_b, du_b);

            // Accelerations should be equal and opposite
            REQUIRE_FLOAT_EQUAL_NAMED("dv_x antisymmetric", dv_a[0], -dv_b[0], 1e-12);
            REQUIRE_FLOAT_EQUAL_NAMED("dv_y zero", dv_a[1], 0.0, 1e-12);
            REQUIRE_FLOAT_EQUAL_NAMED("dv_z zero", dv_a[2], 0.0, 1e-12);
        }
    }

    //==========================================================================
    // SCENARIO: Force scales linearly with p_star
    //==========================================================================

    void test_force_linear_pstar_scaling() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal rho    = 1.0;
        const Tscal omega  = 1.0;
        const Tscal v_star = 0.0;
        const Tscal Fab    = 10.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};
        Tvec v_a           = Tvec{0, 0, 0};

        // Test scaling factors
        const std::vector<Tscal> scale_factors = {0.5, 1.0, 2.0, 5.0, 10.0};
        const Tscal p_star_base                = 1.0;

        Tvec dv_base  = Tvec{0, 0, 0};
        Tscal du_base = 0;
        add_gsph_force_contribution<Tvec, Tscal>(
            m,
            p_star_base,
            v_star,
            rho,
            rho,
            omega,
            omega,
            Fab,
            Fab,
            r_ab_unit,
            v_a,
            dv_base,
            du_base);

        for (Tscal scale : scale_factors) {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m,
                p_star_base * scale,
                v_star,
                rho,
                rho,
                omega,
                omega,
                Fab,
                Fab,
                r_ab_unit,
                v_a,
                dv,
                du);

            // Force should scale linearly with p_star
            REQUIRE_FLOAT_EQUAL_NAMED("force scales with p_star", dv[0], dv_base[0] * scale, 1e-10);
        }
    }

    //==========================================================================
    // SCENARIO: Energy rate is zero when v* = v_a
    //==========================================================================

    void test_energy_rate_zero_when_comoving() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal rho    = 1.0;
        const Tscal omega  = 1.0;
        const Tscal p_star = 1.0;
        const Tscal Fab    = 10.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};

        // Test with particle moving at same velocity as interface
        const std::vector<Tscal> velocities = {0.0, 0.5, 1.0, -0.5};

        for (Tscal v : velocities) {
            Tvec v_a     = Tvec{v, 0, 0};
            Tscal v_star = v; // Interface moving at same speed

            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;

            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            // No work done when co-moving
            REQUIRE_FLOAT_EQUAL_NAMED("zero energy rate for co-moving", du, 0.0, 1e-12);
        }
    }

    //==========================================================================
    // SCENARIO: Force handles zero density safely (boundary particles)
    //==========================================================================

    void test_zero_density_handling() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal omega  = 1.0;
        const Tscal p_star = 1.0;
        const Tscal v_star = 0.5;
        const Tscal Fab    = 10.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};
        Tvec v_a           = Tvec{0, 0, 0};

        // Zero density on one side
        {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, 0.0, 1.0, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            REQUIRE_NAMED("dv_x finite (rho_a=0)", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite (rho_a=0)", std::isfinite(du));
        }

        // Zero density on other side
        {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, 1.0, 0.0, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            REQUIRE_NAMED("dv_x finite (rho_b=0)", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite (rho_b=0)", std::isfinite(du));
        }

        // Both zero density
        {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, 0.0, 0.0, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            REQUIRE_NAMED("dv_x finite (both rho=0)", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite (both rho=0)", std::isfinite(du));
        }
    }

    //==========================================================================
    // SCENARIO: Force handles zero omega safely
    //==========================================================================

    void test_zero_omega_handling() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal rho    = 1.0;
        const Tscal p_star = 1.0;
        const Tscal v_star = 0.5;
        const Tscal Fab    = 10.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};
        Tvec v_a           = Tvec{0, 0, 0};

        // Zero omega on one side
        {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, 0.0, 1.0, Fab, Fab, r_ab_unit, v_a, dv, du);

            REQUIRE_NAMED("dv_x finite (omega_a=0)", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite (omega_a=0)", std::isfinite(du));
        }

        // Both zero omega
        {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, 0.0, 0.0, Fab, Fab, r_ab_unit, v_a, dv, du);

            REQUIRE_NAMED("dv_x finite (both omega=0)", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite (both omega=0)", std::isfinite(du));
        }
    }

    //==========================================================================
    // SCENARIO: Force direction depends on pressure gradient
    //==========================================================================

    void test_force_direction() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal omega  = 1.0;
        const Tscal v_star = 0.0;
        const Tscal Fab    = 10.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};
        Tvec v_a           = Tvec{0, 0, 0};

        // SPH pressure force is always repulsive: particle a is pushed AWAY from neighbor b.
        // With r_ab_unit pointing toward +x (b is to the right of a), particle a
        // accelerates in the -x direction (to the left, away from b).
        // This is independent of which side has higher density - the force is always repulsive.

        // Case 1: Higher density on left
        {
            const Tscal rho_a  = 1.0;
            const Tscal rho_b  = 0.125;
            const Tscal p_star = 0.5;

            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho_a, rho_b, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            // Force is repulsive: a accelerates away from b (negative x)
            REQUIRE_NAMED("repulsive force: a accelerates away from b", dv[0] < 0);
        }

        // Case 2: Lower density on left - force is still repulsive
        {
            const Tscal rho_a  = 0.125;
            const Tscal rho_b  = 1.0;
            const Tscal p_star = 0.5;

            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho_a, rho_b, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            // Force is repulsive: a accelerates away from b (negative x)
            REQUIRE_NAMED("repulsive force: a accelerates away from b", dv[0] < 0);
        }
    }

    //==========================================================================
    // SCENARIO: Complete GSPH pair interaction with Riemann solver
    //==========================================================================

    void test_complete_gsph_interaction() {
        using Tvec  = f64_3;
        using Tscal = f64;

        // Sod shock tube initial conditions
        const Tscal rho_L   = 1.0;
        const Tscal P_L     = 1.0;
        const Tscal u_L     = 0.0;
        const Tscal omega_L = 1.0;

        const Tscal rho_R   = 0.125;
        const Tscal P_R     = 0.1;
        const Tscal u_R     = 0.0;
        const Tscal omega_R = 1.0;

        const Tscal gamma = 1.4;
        const Tscal m     = 1.0;
        const Tscal Fab   = 10.0;
        Tvec r_ab_unit    = Tvec{1, 0, 0};

        // Solve Riemann problem
        auto riemann = riemann::hll_solver<Tscal>(u_L, rho_L, P_L, u_R, rho_R, P_R, gamma);

        // Compute forces
        Tvec dv_L  = Tvec{0, 0, 0};
        Tscal du_L = 0;
        Tvec v_L   = Tvec{u_L, 0, 0};

        add_gsph_force_contribution<Tvec, Tscal>(
            m,
            riemann.p_star,
            riemann.v_star,
            rho_L,
            rho_R,
            omega_L,
            omega_R,
            Fab,
            Fab,
            r_ab_unit,
            v_L,
            dv_L,
            du_L);

        Tvec dv_R  = Tvec{0, 0, 0};
        Tscal du_R = 0;
        Tvec v_R   = Tvec{u_R, 0, 0};

        add_gsph_force_contribution<Tvec, Tscal>(
            m,
            riemann.p_star,
            riemann.v_star,
            rho_R,
            rho_L,
            omega_R,
            omega_L,
            Fab,
            Fab,
            -r_ab_unit,
            v_R,
            dv_R,
            du_R);

        // Physical expectations for SPH pair interaction:
        // 1. Pressure force is repulsive: particles accelerate AWAY from each other
        //    - L accelerates left (negative x, away from R)
        //    - R accelerates right (positive x, away from L)
        REQUIRE_NAMED("L particle accelerates left (away from R)", dv_L[0] < 0);
        REQUIRE_NAMED("R particle accelerates right (away from L)", dv_R[0] > 0);

        // 2. Newton's 3rd law: forces are equal and opposite
        //    With equal masses, accelerations have equal magnitude
        REQUIRE_FLOAT_EQUAL_NAMED(
            "Newton 3rd law: equal magnitude", std::abs(dv_L[0]), std::abs(dv_R[0]), 1e-10);

        // 3. Results are finite
        REQUIRE_NAMED("dv_L finite", std::isfinite(dv_L[0]));
        REQUIRE_NAMED("dv_R finite", std::isfinite(dv_R[0]));
        REQUIRE_NAMED("du_L finite", std::isfinite(du_L));
        REQUIRE_NAMED("du_R finite", std::isfinite(du_R));
    }

    //==========================================================================
    // SCENARIO: Force in different directions
    //==========================================================================

    void test_force_different_directions() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal rho    = 1.0;
        const Tscal omega  = 1.0;
        const Tscal p_star = 1.0;
        const Tscal v_star = 0.0;
        const Tscal Fab    = 10.0;
        Tvec v_a           = Tvec{0, 0, 0};

        // Test different pair directions
        const std::vector<Tvec> directions = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {-1, 0, 0},
            {0, -1, 0},
            {0, 0, -1},
            {1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}, // Diagonal
        };

        for (const auto &dir : directions) {
            Tvec r_ab_unit = dir;
            Tvec dv        = Tvec{0, 0, 0};
            Tscal du       = 0;

            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, omega, omega, Fab, Fab, r_ab_unit, v_a, dv, du);

            // Force should be along the pair direction
            Tscal force_mag = sycl::length(dv);
            Tvec force_dir  = dv / (force_mag + 1e-30);

            // Dot product with direction should be close to 1 or -1
            Tscal alignment = std::abs(sycl::dot(force_dir, r_ab_unit));
            REQUIRE_FLOAT_EQUAL_NAMED("force aligned with pair axis", alignment, 1.0, 0.01);
        }
    }

    //==========================================================================
    // SCENARIO: Force with different kernel values
    //==========================================================================

    void test_force_different_kernels() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tscal m      = 1.0;
        const Tscal rho    = 1.0;
        const Tscal omega  = 1.0;
        const Tscal p_star = 1.0;
        const Tscal v_star = 0.0;
        Tvec r_ab_unit     = Tvec{1, 0, 0};
        Tvec v_a           = Tvec{0, 0, 0};

        // Test different kernel gradient magnitudes
        const std::vector<std::tuple<Tscal, Tscal>> kernel_pairs = {
            {10.0, 10.0},   // Same kernel
            {10.0, 5.0},    // Different kernels
            {5.0, 10.0},    // Reversed
            {1.0, 1.0},     // Small kernel
            {100.0, 100.0}, // Large kernel
        };

        for (const auto &[Fab_a, Fab_b] : kernel_pairs) {
            Tvec dv  = Tvec{0, 0, 0};
            Tscal du = 0;

            add_gsph_force_contribution<Tvec, Tscal>(
                m, p_star, v_star, rho, rho, omega, omega, Fab_a, Fab_b, r_ab_unit, v_a, dv, du);

            // Results should be finite
            REQUIRE_NAMED("dv finite with different kernels", std::isfinite(dv[0]));
            REQUIRE_NAMED("du finite with different kernels", std::isfinite(du));

            // Force should be proportional to sum of kernel gradients
            Tvec dv_ref  = Tvec{0, 0, 0};
            Tscal du_ref = 0;
            add_gsph_force_contribution<Tvec, Tscal>(
                m,
                p_star,
                v_star,
                rho,
                rho,
                omega,
                omega,
                1.0,
                1.0,
                r_ab_unit,
                v_a,
                dv_ref,
                du_ref);

            // Ratio should be approximately (Fab_a + Fab_b) / 2
            Tscal expected_ratio = (Fab_a + Fab_b) / 2.0;
            Tscal actual_ratio   = std::abs(dv[0]) / std::abs(dv_ref[0]);
            REQUIRE_FLOAT_EQUAL_NAMED(
                "force scales with kernel sum", actual_ratio, expected_ratio, 0.01);
        }
    }

} // anonymous namespace

//==============================================================================
// Test registrations
//==============================================================================

TestStart(Unittest, "shammodels/gsph/force/newtons_third", test_gsph_newton3, 1) {
    test_sph_force_newtons_third_law();
}

TestStart(Unittest, "shammodels/gsph/force/symmetry", test_gsph_force_sym, 1) {
    test_gsph_force_symmetry();
}

TestStart(Unittest, "shammodels/gsph/force/pstar_scaling", test_gsph_pstar, 1) {
    test_force_linear_pstar_scaling();
}

TestStart(Unittest, "shammodels/gsph/force/energy_comoving", test_gsph_energy_como, 1) {
    test_energy_rate_zero_when_comoving();
}

TestStart(Unittest, "shammodels/gsph/force/zero_density", test_gsph_zero_rho, 1) {
    test_zero_density_handling();
}

TestStart(Unittest, "shammodels/gsph/force/zero_omega", test_gsph_zero_omega, 1) {
    test_zero_omega_handling();
}

TestStart(Unittest, "shammodels/gsph/force/direction", test_gsph_force_dir, 1) {
    test_force_direction();
}

TestStart(Unittest, "shammodels/gsph/force/complete_interaction", test_gsph_complete, 1) {
    test_complete_gsph_interaction();
}

TestStart(Unittest, "shammodels/gsph/force/different_directions", test_gsph_dirs, 1) {
    test_force_different_directions();
}

TestStart(Unittest, "shammodels/gsph/force/different_kernels", test_gsph_kernels, 1) {
    test_force_different_kernels();
}
