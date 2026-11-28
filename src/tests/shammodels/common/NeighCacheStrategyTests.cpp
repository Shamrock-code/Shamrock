// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighCacheStrategyTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Unit tests for NeighCacheStrategy JSON (de)serialization and backward compatibility
 */

#include "shammodels/common/config/enum_NeighCacheStrategy.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamtest/shamtest.hpp"

namespace {

    using shammodels::NeighCacheStrategy;

    using SPHConfig  = shammodels::sph::SolverConfig<f64_3, shammath::M4>;
    using GSPHConfig = shammodels::gsph::SolverConfig<f64_3, shammath::M4>;

    /// REQUIRE_EQUAL needs a formattable type, so strategies are compared through their json name
    std::string strategy_name(NeighCacheStrategy strategy) {
        return nlohmann::json(strategy).get<std::string>();
    }

    //==========================================================================
    // SCENARIO: NeighCacheStrategy JSON roundtrip
    //==========================================================================

    void test_neigh_cache_strategy_json_roundtrip() {
        for (NeighCacheStrategy strategy :
             {NeighCacheStrategy::SingleStage, NeighCacheStrategy::TwoStage}) {

            nlohmann::json j = strategy;
            REQUIRE(j.is_string());

            NeighCacheStrategy out = nlohmann::json::parse(j.dump()).get<NeighCacheStrategy>();
            REQUIRE_EQUAL(strategy_name(out), strategy_name(strategy));
        }
    }

    void test_neigh_cache_strategy_json_names() {
        REQUIRE_EQUAL(strategy_name(NeighCacheStrategy::SingleStage), std::string("single_stage"));
        REQUIRE_EQUAL(strategy_name(NeighCacheStrategy::TwoStage), std::string("two_stage"));
    }

    void test_neigh_cache_strategy_json_unknown_throws() {
        nlohmann::json j = "three_stage";
        REQUIRE_EXCEPTION_THROW(j.get<NeighCacheStrategy>(), std::runtime_error);
    }

    //==========================================================================
    // SCENARIO: solver configs carry the strategy through json
    //==========================================================================

    /// The solver config json must use the new key, and not the legacy boolean one
    template<class Config>
    void test_config_json_roundtrip() {
        for (NeighCacheStrategy strategy :
             {NeighCacheStrategy::SingleStage, NeighCacheStrategy::TwoStage}) {

            Config in_cfg;
            in_cfg.set_neigh_cache_strategy(strategy);

            nlohmann::json j = in_cfg;
            REQUIRE(j.contains(shammodels::neigh_cache_strategy_json_key));
            REQUIRE(!j.contains(shammodels::neigh_cache_strategy_legacy_json_key));

            Config out_cfg = nlohmann::json::parse(j.dump(4)).template get<Config>();
            REQUIRE_EQUAL(strategy_name(out_cfg.neigh_cache_strategy), strategy_name(strategy));
        }
    }

    /// The default must stay the two stage search, as it was when it was a boolean
    template<class Config>
    void test_config_default_is_two_stage() {
        Config cfg;
        REQUIRE_EQUAL(
            strategy_name(cfg.neigh_cache_strategy), strategy_name(NeighCacheStrategy::TwoStage));
    }

    /// A config dumped before the strategy became an enum must still load
    template<class Config>
    void test_config_json_legacy_bool_fallback() {
        for (bool legacy_value : {false, true}) {

            Config in_cfg;
            nlohmann::json j = in_cfg;

            j.erase(shammodels::neigh_cache_strategy_json_key);
            j[shammodels::neigh_cache_strategy_legacy_json_key] = legacy_value;

            Config out_cfg = j.template get<Config>();

            REQUIRE_EQUAL(
                strategy_name(out_cfg.neigh_cache_strategy),
                strategy_name(
                    (legacy_value) ? NeighCacheStrategy::TwoStage
                                   : NeighCacheStrategy::SingleStage));
        }
    }

    /// If both keys are there the new one wins
    template<class Config>
    void test_config_json_new_key_wins_over_legacy() {
        Config in_cfg;
        in_cfg.set_neigh_cache_strategy(NeighCacheStrategy::SingleStage);

        nlohmann::json j                                    = in_cfg;
        j[shammodels::neigh_cache_strategy_legacy_json_key] = true;

        Config out_cfg = j.template get<Config>();
        REQUIRE_EQUAL(
            strategy_name(out_cfg.neigh_cache_strategy),
            strategy_name(NeighCacheStrategy::SingleStage));
    }

    /// A config json from before the strategy existed at all keeps the default
    template<class Config>
    void test_config_json_missing_key_keeps_default() {
        Config in_cfg;
        nlohmann::json j = in_cfg;
        j.erase(shammodels::neigh_cache_strategy_json_key);

        Config out_cfg = j.template get<Config>();
        REQUIRE_EQUAL(
            strategy_name(out_cfg.neigh_cache_strategy),
            strategy_name(NeighCacheStrategy::TwoStage));
    }

    /// The deprecated boolean setter must keep mapping onto the enum
    template<class Config>
    void test_config_deprecated_bool_setter() {
        Config cfg;

        cfg.set_two_stage_search(false);
        REQUIRE_EQUAL(
            strategy_name(cfg.neigh_cache_strategy),
            strategy_name(NeighCacheStrategy::SingleStage));

        cfg.set_two_stage_search(true);
        REQUIRE_EQUAL(
            strategy_name(cfg.neigh_cache_strategy), strategy_name(NeighCacheStrategy::TwoStage));
    }

} // namespace

NEW_TEST(Unittest, "shammodels/common/config/neigh_cache_strategy_json_roundtrip", 1) {
    test_neigh_cache_strategy_json_roundtrip();
}

NEW_TEST(Unittest, "shammodels/common/config/neigh_cache_strategy_json_names", 1) {
    test_neigh_cache_strategy_json_names();
}

NEW_TEST(Unittest, "shammodels/common/config/neigh_cache_strategy_json_unknown_throws", 1) {
    test_neigh_cache_strategy_json_unknown_throws();
}

NEW_TEST(Unittest, "shammodels/sph/config/neigh_cache_strategy_json", 1) {
    test_config_default_is_two_stage<SPHConfig>();
    test_config_json_roundtrip<SPHConfig>();
    test_config_json_legacy_bool_fallback<SPHConfig>();
    test_config_json_new_key_wins_over_legacy<SPHConfig>();
    test_config_json_missing_key_keeps_default<SPHConfig>();
    test_config_deprecated_bool_setter<SPHConfig>();
}

NEW_TEST(Unittest, "shammodels/gsph/config/neigh_cache_strategy_json", 1) {
    test_config_default_is_two_stage<GSPHConfig>();
    test_config_json_roundtrip<GSPHConfig>();
    test_config_json_legacy_bool_fallback<GSPHConfig>();
    test_config_json_new_key_wins_over_legacy<GSPHConfig>();
    test_config_json_missing_key_keeps_default<GSPHConfig>();
    test_config_deprecated_bool_setter<GSPHConfig>();
}
