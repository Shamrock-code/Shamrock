// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHConfigJsonTests.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Unit tests for ForceFormulationConfig and RiemannConfig JSON (de)serialization
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/config/ForceFormulationConfig.hpp"
#include "shammodels/gsph/config/RiemannConfig.hpp"
#include "shamtest/shamtest.hpp"

namespace {

    //==========================================================================
    // SCENARIO: ForceFormulationConfig JSON roundtrip
    //==========================================================================

    void test_force_formulation_json_cha_whitworth() {
        using Config = shammodels::gsph::ForceFormulationConfig<f64_3>;

        Config in_cfg;
        in_cfg.set_cha_whitworth();
        in_cfg.print_status();

        nlohmann::json j    = in_cfg;
        nlohmann::json jout = nlohmann::json::parse(j.dump(4));
        Config out_cfg      = jout.template get<Config>();

        REQUIRE(out_cfg.is_cha_whitworth());
        REQUIRE(!out_cfg.is_inutsuka_v2());
    }

    void test_force_formulation_json_inutsuka_v2() {
        using Config = shammodels::gsph::ForceFormulationConfig<f64_3>;

        Config in_cfg;
        in_cfg.set_inutsuka_v2();
        in_cfg.print_status();

        nlohmann::json j    = in_cfg;
        nlohmann::json jout = nlohmann::json::parse(j.dump(4));
        Config out_cfg      = jout.template get<Config>();

        REQUIRE(out_cfg.is_inutsuka_v2());
        REQUIRE(!out_cfg.is_cha_whitworth());
    }

    void test_force_formulation_json_unknown_type_throws() {
        using Config = shammodels::gsph::ForceFormulationConfig<f64_3>;

        nlohmann::json j = {{"force_formulation", "not_a_real_formulation"}};
        REQUIRE_EXCEPTION_THROW(j.template get<Config>(), std::runtime_error);
    }

    //==========================================================================
    // SCENARIO: RiemannConfig::Exact JSON roundtrip and backward compatibility
    //==========================================================================

    void test_riemann_config_exact_json_roundtrip() {
        using Config = shammodels::gsph::RiemannConfig<f64_3>;

        Config in_cfg;
        in_cfg.set_exact(1e-9, 42);
        in_cfg.print_status();

        nlohmann::json j    = in_cfg;
        nlohmann::json jout = nlohmann::json::parse(j.dump(4));
        Config out_cfg      = jout.template get<Config>();

        REQUIRE(out_cfg.is_exact());
        using Exact = typename Config::Exact;
        if (const Exact *v = std::get_if<Exact>(&out_cfg.config)) {
            REQUIRE_FLOAT_EQUAL_NAMED("tol roundtrips", v->tol, f64(1e-9), 1e-15);
            REQUIRE_EQUAL_NAMED("max_iter roundtrips", v->max_iter, 42);
        } else {
            REQUIRE(false);
        }
    }

    void test_riemann_config_exact_json_backward_compat_missing_max_iter() {
        using Config = shammodels::gsph::RiemannConfig<f64_3>;
        using Exact  = typename Config::Exact;

        // Simulates a config saved before max_iter existed (tol-only).
        nlohmann::json j = {{"riemann_type", "exact"}, {"tol", 1e-7}};

        Config out_cfg = j.template get<Config>();

        REQUIRE(out_cfg.is_exact());
        if (const Exact *v = std::get_if<Exact>(&out_cfg.config)) {
            REQUIRE_FLOAT_EQUAL_NAMED("tol reads back", v->tol, f64(1e-7), 1e-15);
            REQUIRE_EQUAL_NAMED("max_iter falls back to default", v->max_iter, Exact{}.max_iter);
        } else {
            REQUIRE(false);
        }
    }

    void test_riemann_config_iterative_and_hllc_json_roundtrip() {
        using Config = shammodels::gsph::RiemannConfig<f64_3>;

        Config iter_cfg;
        iter_cfg.set_iterative(1e-5, 15);
        nlohmann::json j_iter = iter_cfg;
        Config iter_out       = nlohmann::json::parse(j_iter.dump()).template get<Config>();
        using Iterative       = typename Config::Iterative;
        REQUIRE(iter_out.is_iterative());
        if (const Iterative *v = std::get_if<Iterative>(&iter_out.config)) {
            REQUIRE_FLOAT_EQUAL_NAMED("iterative tol roundtrips", v->tol, f64(1e-5), 1e-15);
            REQUIRE_EQUAL_NAMED("iterative max_iter roundtrips", v->max_iter, 15);
        } else {
            REQUIRE(false);
        }

        Config hllc_cfg;
        hllc_cfg.set_hllc();
        nlohmann::json j_hllc = hllc_cfg;
        Config hllc_out       = nlohmann::json::parse(j_hllc.dump()).template get<Config>();
        REQUIRE(hllc_out.is_hllc());
    }

    //==========================================================================
    // SCENARIO: full SolverConfig JSON roundtrip, including force_formulation_config
    //==========================================================================

    void test_solver_config_json_roundtrip_with_force_formulation() {
        using Config = shammodels::gsph::SolverConfig<f64_3, shammath::M4>;

        Config in_cfg;
        in_cfg.set_force_inutsuka_v2();
        in_cfg.set_riemann_exact(1e-9, 42);
        in_cfg.set_eos_adiabatic(1.4);
        in_cfg.gpart_mass = 2.5;
        in_cfg.print_status();

        nlohmann::json j    = in_cfg;
        nlohmann::json jout = nlohmann::json::parse(j.dump(4));
        Config out_cfg      = jout.template get<Config>();

        REQUIRE(out_cfg.is_force_inutsuka_v2());
        REQUIRE(out_cfg.riemann_config.is_exact());
        REQUIRE_FLOAT_EQUAL_NAMED("gpart_mass roundtrips", out_cfg.gpart_mass, f64(2.5), 1e-15);
        out_cfg.check_config(); // InutsukaV2 + Exact must still be a valid combination
    }

} // namespace

NEW_TEST(Unittest, "shammodels/gsph/config/force_formulation_json_cha_whitworth", 1) {
    test_force_formulation_json_cha_whitworth();
}

NEW_TEST(Unittest, "shammodels/gsph/config/force_formulation_json_inutsuka_v2", 1) {
    test_force_formulation_json_inutsuka_v2();
}

NEW_TEST(Unittest, "shammodels/gsph/config/force_formulation_json_unknown_throws", 1) {
    test_force_formulation_json_unknown_type_throws();
}

NEW_TEST(Unittest, "shammodels/gsph/config/riemann_exact_json_roundtrip", 1) {
    test_riemann_config_exact_json_roundtrip();
}

NEW_TEST(Unittest, "shammodels/gsph/config/riemann_exact_json_backward_compat", 1) {
    test_riemann_config_exact_json_backward_compat_missing_max_iter();
}

NEW_TEST(Unittest, "shammodels/gsph/config/riemann_iterative_hllc_json_roundtrip", 1) {
    test_riemann_config_iterative_and_hllc_json_roundtrip();
}

NEW_TEST(Unittest, "shammodels/gsph/config/solver_config_json_roundtrip", 1) {
    test_solver_config_json_roundtrip_with_force_formulation();
}
