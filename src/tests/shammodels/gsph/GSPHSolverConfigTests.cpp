// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHSolverConfigTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Unit tests for GSPH SolverConfig::check_config() validation
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shamtest/shamtest.hpp"

namespace {

    using Config = shammodels::gsph::SolverConfig<f64_3, shammath::M4>;

    //==========================================================================
    // SCENARIO: InutsukaV2 + HLLC is rejected (not wired into update_derivs_hllc)
    //==========================================================================

    void test_inutsuka_v2_hllc_rejected() {
        Config cfg;
        cfg.set_force_inutsuka_v2();
        cfg.set_riemann_hllc();
        cfg.set_eos_adiabatic(1.4);

        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::runtime_error);
    }

    //==========================================================================
    // SCENARIO: InutsukaV2 with the iterative or exact solver is accepted
    //==========================================================================

    void test_inutsuka_v2_iterative_and_exact_accepted() {
        Config cfg_iter;
        cfg_iter.set_force_inutsuka_v2();
        cfg_iter.set_riemann_iterative();
        cfg_iter.set_eos_adiabatic(1.4);
        cfg_iter.check_config(); // must not throw

        Config cfg_exact;
        cfg_exact.set_force_inutsuka_v2();
        cfg_exact.set_riemann_exact();
        cfg_exact.set_eos_adiabatic(1.4);
        cfg_exact.check_config(); // must not throw

        REQUIRE(true);
    }

    //==========================================================================
    // SCENARIO: ChaWhitworth (default force formulation) with HLLC is accepted
    //==========================================================================

    void test_cha_whitworth_hllc_accepted() {
        Config cfg;
        cfg.set_riemann_hllc();
        cfg.set_eos_adiabatic(1.4);
        cfg.check_config(); // must not throw

        REQUIRE(true);
    }

    //==========================================================================
    // SCENARIO: set_force_cha_whitworth() and is_force_inutsuka_v2() wrappers
    //==========================================================================

    void test_set_force_cha_whitworth_explicit() {
        Config cfg;
        cfg.set_force_inutsuka_v2();
        REQUIRE(cfg.is_force_inutsuka_v2());

        // Explicitly switch back, exercising the setter directly rather than
        // relying on the default-constructed value.
        cfg.set_force_cha_whitworth();
        REQUIRE(!cfg.is_force_inutsuka_v2());
    }

    //==========================================================================
    // SCENARIO: SolverConfig::print_status() runs without throwing, for either
    // force formulation
    //==========================================================================

    void test_print_status_smoke() {
        Config cfg_cha;
        cfg_cha.set_force_cha_whitworth();
        cfg_cha.set_eos_adiabatic(1.4);
        cfg_cha.print_status();

        Config cfg_v2;
        cfg_v2.set_force_inutsuka_v2();
        cfg_v2.set_riemann_exact();
        cfg_v2.set_eos_adiabatic(1.4);
        cfg_v2.print_status();

        REQUIRE(true);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/gsph/solverconfig/inutsuka_v2_hllc_rejected", 1) {
    test_inutsuka_v2_hllc_rejected();
}

NEW_TEST(Unittest, "shammodels/gsph/solverconfig/inutsuka_v2_iterative_exact_accepted", 1) {
    test_inutsuka_v2_iterative_and_exact_accepted();
}

NEW_TEST(Unittest, "shammodels/gsph/solverconfig/cha_whitworth_hllc_accepted", 1) {
    test_cha_whitworth_hllc_accepted();
}

NEW_TEST(Unittest, "shammodels/gsph/solverconfig/set_force_cha_whitworth", 1) {
    test_set_force_cha_whitworth_explicit();
}

NEW_TEST(Unittest, "shammodels/gsph/solverconfig/print_status_smoke", 1) {
    test_print_status_smoke();
}
