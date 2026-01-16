// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHIntegrationTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Comprehensive BDD-style unit tests for GSPH time integration
 *
 * Tests cover:
 * - Leapfrog predictor-corrector scheme (KDK)
 * - Second order convergence verification
 * - Energy conservation for symplectic integrators
 * - Internal energy integration
 * - Edge cases (zero acceleration, constant force)
 */

#include "shamtest/shamtest.hpp"
#include <cmath>
#include <vector>

namespace {

    //==========================================================================
    // Leapfrog KDK integration kernels (extracted for testing)
    //==========================================================================

    template<class Tscal, class Tvec>
    void predictor_step(Tvec &x, Tvec &v, const Tvec &a_old, Tscal dt) {
        Tscal half_dt = dt / Tscal{2};
        v             = v + a_old * half_dt;
        x             = x + v * dt;
    }

    template<class Tscal, class Tvec>
    void corrector_step(Tvec &v, const Tvec &a_new, Tscal dt) {
        Tscal half_dt = dt / Tscal{2};
        v             = v + a_new * half_dt;
    }

    template<class Tscal>
    void predictor_step_energy(Tscal &u, const Tscal &du_old, Tscal dt) {
        Tscal half_dt = dt / Tscal{2};
        u             = u + du_old * half_dt;
    }

    template<class Tscal>
    void corrector_step_energy(Tscal &u, const Tscal &du_new, Tscal dt) {
        Tscal half_dt = dt / Tscal{2};
        u             = u + du_new * half_dt;
    }

    //==========================================================================
    // SCENARIO: Leapfrog is exact for constant acceleration
    //==========================================================================

    void test_constant_acceleration_1d() {
        using Tscal = f64;

        Tscal x = 0.0;
        Tscal v = 1.0;
        Tscal a = -10.0; // Constant acceleration

        const Tscal dt    = 0.01;
        const u32 n_steps = 100;
        const Tscal t_end = n_steps * dt;

        for (u32 i = 0; i < n_steps; ++i) {
            Tscal half_dt = dt / 2;
            v             = v + a * half_dt;
            x             = x + v * dt;
            v             = v + a * half_dt;
        }

        // Exact solution: x = x0 + v0*t + 0.5*a*t^2
        Tscal x_exact = 0.0 + 1.0 * t_end + 0.5 * a * t_end * t_end;
        Tscal v_exact = 1.0 + a * t_end;

        REQUIRE_FLOAT_EQUAL_NAMED("x exact for constant a", x, x_exact, 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("v exact for constant a", v, v_exact, 1e-10);
    }

    void test_constant_acceleration_3d() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const Tvec x0 = {0, 0, 0};   // Initial position
        const Tvec v0 = {1, 2, 3};   // Initial velocity
        const Tvec a  = {0, -10, 5}; // Constant acceleration

        Tvec x = x0;
        Tvec v = v0;

        const Tscal dt    = 0.01;
        const u32 n_steps = 100;
        const Tscal t_end = n_steps * dt;

        for (u32 i = 0; i < n_steps; ++i) {
            predictor_step<Tscal, Tvec>(x, v, a, dt);
            corrector_step<Tscal, Tvec>(v, a, dt);
        }

        // Exact solution: x = x0 + v0*t + 0.5*a*t^2, v = v0 + a*t
        Tvec x_exact;
        x_exact[0] = x0[0] + v0[0] * t_end + Tscal{0.5} * a[0] * t_end * t_end;
        x_exact[1] = x0[1] + v0[1] * t_end + Tscal{0.5} * a[1] * t_end * t_end;
        x_exact[2] = x0[2] + v0[2] * t_end + Tscal{0.5} * a[2] * t_end * t_end;

        Tvec v_exact;
        v_exact[0] = v0[0] + a[0] * t_end;
        v_exact[1] = v0[1] + a[1] * t_end;
        v_exact[2] = v0[2] + a[2] * t_end;

        REQUIRE_FLOAT_EQUAL_NAMED("x[0] exact", x[0], x_exact[0], 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("x[1] exact", x[1], x_exact[1], 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("x[2] exact", x[2], x_exact[2], 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("v[0] exact", v[0], v_exact[0], 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("v[1] exact", v[1], v_exact[1], 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("v[2] exact", v[2], v_exact[2], 1e-10);
    }

    //==========================================================================
    // SCENARIO: Leapfrog is 2nd order for harmonic oscillator
    //==========================================================================

    void test_harmonic_oscillator() {
        using Tscal = f64;

        const Tscal omega  = 2.0 * M_PI;
        const Tscal omega2 = omega * omega;

        Tscal x = 1.0;
        Tscal v = 0.0;

        const Tscal dt    = 0.001;
        const Tscal t_end = 1.0;
        const u32 n_steps = static_cast<u32>(t_end / dt);

        for (u32 i = 0; i < n_steps; ++i) {
            Tscal a_old   = -omega2 * x;
            Tscal half_dt = dt / 2;

            v = v + a_old * half_dt;
            x = x + v * dt;

            Tscal a_new = -omega2 * x;
            v           = v + a_new * half_dt;
        }

        // After one period, should return to initial state
        REQUIRE_FLOAT_EQUAL_NAMED("x returns to start", x, 1.0, 0.01);
        REQUIRE_FLOAT_EQUAL_NAMED("v returns to start", v, 0.0, 0.1);
    }

    void test_harmonic_oscillator_energy() {
        using Tscal = f64;

        const Tscal omega2 = 1.0;
        const Tscal m      = 1.0;

        Tscal x = 1.0;
        Tscal v = 0.0;

        const Tscal E0 = 0.5 * m * v * v + 0.5 * omega2 * x * x;

        const Tscal dt    = 0.01;
        const u32 n_steps = 1000;

        Tscal max_energy_error = 0;

        for (u32 i = 0; i < n_steps; ++i) {
            Tscal a_old   = -omega2 * x;
            Tscal half_dt = dt / 2;

            v = v + a_old * half_dt;
            x = x + v * dt;

            Tscal a_new = -omega2 * x;
            v           = v + a_new * half_dt;

            Tscal E            = 0.5 * m * v * v + 0.5 * omega2 * x * x;
            Tscal energy_error = std::abs(E - E0) / E0;
            max_energy_error   = std::max(max_energy_error, energy_error);
        }

        // Symplectic integrators have bounded energy error
        REQUIRE_NAMED("energy bounded", max_energy_error < 0.01);
    }

    //==========================================================================
    // SCENARIO: Convergence order is 2nd order
    //==========================================================================

    void test_convergence_order() {
        using Tscal = f64;

        const Tscal omega2 = 1.0;
        const Tscal t_end  = 1.0;

        auto integrate = [omega2, t_end](Tscal dt) -> Tscal {
            Tscal x       = 1.0;
            Tscal v       = 0.0;
            u32 n_steps   = static_cast<u32>(t_end / dt);
            Tscal half_dt = dt / 2;

            for (u32 i = 0; i < n_steps; ++i) {
                Tscal a_old = -omega2 * x;
                v           = v + a_old * half_dt;
                x           = x + v * dt;
                Tscal a_new = -omega2 * x;
                v           = v + a_new * half_dt;
            }

            Tscal x_exact = std::cos(t_end);
            return std::abs(x - x_exact);
        };

        Tscal dt1    = 0.01;
        Tscal dt2    = 0.005;
        Tscal error1 = integrate(dt1);
        Tscal error2 = integrate(dt2);

        // 2nd order: error ~ dt^2, so halving dt should quarter error
        Tscal ratio = error1 / error2;

        REQUIRE_NAMED("2nd order convergence", ratio > 3.0 && ratio < 5.0);
    }

    //==========================================================================
    // SCENARIO: Internal energy integration
    //==========================================================================

    void test_energy_integration_constant_rate() {
        using Tscal = f64;

        Tscal u  = 1.0;
        Tscal du = 0.5;

        const Tscal dt    = 0.1;
        const u32 n_steps = 10;
        const Tscal t_end = n_steps * dt;

        for (u32 i = 0; i < n_steps; ++i) {
            predictor_step_energy<Tscal>(u, du, dt);
            corrector_step_energy<Tscal>(u, du, dt);
        }

        Tscal u_exact = 1.0 + du * t_end;
        REQUIRE_FLOAT_EQUAL_NAMED("u exact for constant du", u, u_exact, 1e-12);
    }

    void test_energy_integration_varying_rate() {
        using Tscal = f64;

        const Tscal omega = 1.0;
        const Tscal du0   = 1.0;
        const Tscal dt    = 0.01;
        const u32 n_steps = 100;

        Tscal u = 0.0;
        Tscal t = 0.0;

        for (u32 i = 0; i < n_steps; ++i) {
            Tscal du_old = du0 * std::cos(omega * t);
            predictor_step_energy<Tscal>(u, du_old, dt);

            t += dt;

            Tscal du_new = du0 * std::cos(omega * t);
            corrector_step_energy<Tscal>(u, du_new, dt);
        }

        Tscal u_exact = du0 * std::sin(omega * t) / omega;
        REQUIRE_FLOAT_EQUAL_NAMED("u matches integrated cos", u, u_exact, 0.01);
    }

    //==========================================================================
    // SCENARIO: Predictor-corrector uses average acceleration
    //==========================================================================

    void test_predictor_corrector_average() {
        using Tvec  = f64_3;
        using Tscal = f64;

        Tvec x = {0, 0, 0};
        Tvec v = {0, 0, 0};

        Tvec a_old = {1, 0, 0};
        Tvec a_new = {3, 0, 0}; // Different new acceleration

        Tscal dt = 1.0;

        predictor_step<Tscal, Tvec>(x, v, a_old, dt);
        corrector_step<Tscal, Tvec>(v, a_new, dt);

        // v = v0 + 0.5*a_old*dt + 0.5*a_new*dt = 0 + 0.5*1*1 + 0.5*3*1 = 2
        REQUIRE_FLOAT_EQUAL_NAMED("v uses average a", v[0], 2.0, 1e-12);

        // x = x0 + (v0 + a_old*dt/2)*dt = 0 + 0.5*1 = 0.5
        REQUIRE_FLOAT_EQUAL_NAMED("x at midpoint v", x[0], 0.5, 1e-12);
    }

    //==========================================================================
    // SCENARIO: Zero acceleration
    //==========================================================================

    void test_zero_acceleration() {
        using Tvec  = f64_3;
        using Tscal = f64;

        Tvec x = {0, 0, 0};
        Tvec v = {1, 2, 3};
        Tvec a = {0, 0, 0};

        const Tscal dt    = 0.01;
        const u32 n_steps = 100;
        const Tscal t_end = n_steps * dt;

        for (u32 i = 0; i < n_steps; ++i) {
            predictor_step<Tscal, Tvec>(x, v, a, dt);
            corrector_step<Tscal, Tvec>(v, a, dt);
        }

        // With zero acceleration: x = x0 + v0*t, v = v0
        REQUIRE_FLOAT_EQUAL_NAMED("x[0] linear", x[0], 1.0 * t_end, 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("x[1] linear", x[1], 2.0 * t_end, 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("x[2] linear", x[2], 3.0 * t_end, 1e-10);
        REQUIRE_FLOAT_EQUAL_NAMED("v[0] unchanged", v[0], 1.0, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("v[1] unchanged", v[1], 2.0, 1e-12);
        REQUIRE_FLOAT_EQUAL_NAMED("v[2] unchanged", v[2], 3.0, 1e-12);
    }

    //==========================================================================
    // SCENARIO: Large timestep stability
    //==========================================================================

    void test_large_timestep_stability() {
        using Tscal = f64;

        const Tscal omega2 = 1.0;
        const Tscal dt     = 0.5; // Large timestep
        const u32 n_steps  = 100;

        Tscal x = 1.0;
        Tscal v = 0.0;

        Tscal x_max = 0;
        Tscal v_max = 0;

        for (u32 i = 0; i < n_steps; ++i) {
            Tscal a_old   = -omega2 * x;
            Tscal half_dt = dt / 2;

            v = v + a_old * half_dt;
            x = x + v * dt;

            Tscal a_new = -omega2 * x;
            v           = v + a_new * half_dt;

            x_max = std::max(x_max, std::abs(x));
            v_max = std::max(v_max, std::abs(v));
        }

        // Leapfrog is stable for omega*dt < 2
        // Here omega*dt = 0.5 < 2, so should be stable
        REQUIRE_NAMED("x bounded", x_max < 2.0);
        REQUIRE_NAMED("v bounded", v_max < 2.0);
    }

    //==========================================================================
    // SCENARIO: Multiple particles with different accelerations
    //==========================================================================

    void test_multiple_particles() {
        using Tvec  = f64_3;
        using Tscal = f64;

        const u32 n_particles = 10;

        std::vector<Tvec> x(n_particles);
        std::vector<Tvec> v(n_particles);
        std::vector<Tvec> a(n_particles);

        for (u32 i = 0; i < n_particles; ++i) {
            x[i] = Tvec{Tscal(i), 0, 0};
            v[i] = Tvec{0, 0, 0};
            a[i] = Tvec{0, -Tscal(i + 1), 0}; // Different acceleration for each
        }

        const Tscal dt    = 0.01;
        const u32 n_steps = 100;
        const Tscal t_end = n_steps * dt;

        for (u32 step = 0; step < n_steps; ++step) {
            for (u32 i = 0; i < n_particles; ++i) {
                predictor_step<Tscal, Tvec>(x[i], v[i], a[i], dt);
                corrector_step<Tscal, Tvec>(v[i], a[i], dt);
            }
        }

        // Verify each particle followed its trajectory
        for (u32 i = 0; i < n_particles; ++i) {
            Tscal x_expected  = Tscal(i);
            Tscal y_expected  = 0.5 * a[i][1] * t_end * t_end;
            Tscal vy_expected = a[i][1] * t_end;

            REQUIRE_FLOAT_EQUAL_NAMED("particle x unchanged", x[i][0], x_expected, 1e-10);
            REQUIRE_FLOAT_EQUAL_NAMED("particle y correct", x[i][1], y_expected, 1e-8);
            REQUIRE_FLOAT_EQUAL_NAMED("particle vy correct", v[i][1], vy_expected, 1e-10);
        }
    }

} // anonymous namespace

//==============================================================================
// Test registrations
//==============================================================================

TestStart(Unittest, "shammodels/gsph/integration/constant_1d", test_gsph_const1d, 1) {
    test_constant_acceleration_1d();
}

TestStart(Unittest, "shammodels/gsph/integration/constant_3d", test_gsph_const3d, 1) {
    test_constant_acceleration_3d();
}

TestStart(Unittest, "shammodels/gsph/integration/oscillator", test_gsph_osc, 1) {
    test_harmonic_oscillator();
}

TestStart(Unittest, "shammodels/gsph/integration/oscillator_energy", test_gsph_osc_e, 1) {
    test_harmonic_oscillator_energy();
}

TestStart(Unittest, "shammodels/gsph/integration/convergence", test_gsph_conv, 1) {
    test_convergence_order();
}

TestStart(Unittest, "shammodels/gsph/integration/energy_const", test_gsph_u_const, 1) {
    test_energy_integration_constant_rate();
}

TestStart(Unittest, "shammodels/gsph/integration/energy_var", test_gsph_u_var, 1) {
    test_energy_integration_varying_rate();
}

TestStart(Unittest, "shammodels/gsph/integration/average_accel", test_gsph_avg_a, 1) {
    test_predictor_corrector_average();
}

TestStart(Unittest, "shammodels/gsph/integration/zero_accel", test_gsph_zero_a, 1) {
    test_zero_acceleration();
}

TestStart(Unittest, "shammodels/gsph/integration/stability", test_gsph_stable, 1) {
    test_large_timestep_stability();
}

TestStart(Unittest, "shammodels/gsph/integration/multi_particle", test_gsph_multi, 1) {
    test_multiple_particles();
}
