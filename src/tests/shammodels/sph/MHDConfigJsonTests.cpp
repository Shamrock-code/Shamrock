// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MHDConfigJsonTests.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Unit tests for the SPH MHDConfig JSON (de)serialization
 */

#include "shammodels/sph/config/MHDConfig.hpp"
#include "shamtest/shamtest.hpp"

namespace {

    using Config      = shammodels::sph::MHDConfig<f64_3>;
    using NonIdealMHD = typename Config::NonIdealMHD;

    Config roundtrip(const Config &in_cfg) {
        nlohmann::json j    = in_cfg;
        nlohmann::json jout = nlohmann::json::parse(j.dump(4));
        return jout.template get<Config>();
    }

    void test_nimhd_json_roundtrip() {
        for (bool eta_fields : {false, true}) {
            for (bool eta_semi_constant : {false, true}) {
                Config in_cfg;
                in_cfg.set(NonIdealMHD{0.2, 0.5, 1e-3, -2e-2, 3e-3, eta_fields, eta_semi_constant});

                Config out_cfg = roundtrip(in_cfg);

                REQUIRE(out_cfg.do_nimhd());
                if (const NonIdealMHD *v = std::get_if<NonIdealMHD>(&out_cfg.configMHD)) {
                    REQUIRE_EQUAL_NAMED("sigma_mhd", v->sigma_mhd, 0.2);
                    REQUIRE_EQUAL_NAMED("alpha_B", v->alpha_B, 0.5);
                    REQUIRE_EQUAL_NAMED("etaO", v->etaO, 1e-3);
                    REQUIRE_EQUAL_NAMED("etaH", v->etaH, -2e-2);
                    REQUIRE_EQUAL_NAMED("etaAD", v->etaAD, 3e-3);
                    REQUIRE_EQUAL_NAMED("eta_fields", v->eta_fields, eta_fields);
                    REQUIRE_EQUAL_NAMED(
                        "eta_semi_constant", v->eta_semi_constant, eta_semi_constant);
                } else {
                    REQUIRE(false);
                }

                REQUIRE_EQUAL(out_cfg.has_field_eta(), eta_fields || eta_semi_constant);
                REQUIRE_EQUAL(out_cfg.has_eta_semi_constant(), eta_semi_constant);
            }
        }
    }

    void test_nimhd_json_backward_compat() {
        // Simulates a config saved before eta_fields / eta_semi_constant existed, which still
        // carried the artificial viscosity parameters (now ignored).
        nlohmann::json j
            = {{"mhd_type", "non_ideal_mhd"},
               {"sigma_mhd", 0.1},
               {"alpha_u", 1.0},
               {"alpha_B", 1.0},
               {"alpha_AV", 1.0},
               {"beta_AV", 2.0},
               {"etaO", 1e-3},
               {"etaH", 0.0},
               {"etaAD", 0.0}};

        Config out_cfg = j.template get<Config>();

        REQUIRE(out_cfg.do_nimhd());
        if (const NonIdealMHD *v = std::get_if<NonIdealMHD>(&out_cfg.configMHD)) {
            REQUIRE_EQUAL_NAMED("etaO reads back", v->etaO, 1e-3);
            REQUIRE_EQUAL_NAMED("eta_fields defaults", v->eta_fields, NonIdealMHD{}.eta_fields);
            REQUIRE_EQUAL_NAMED(
                "eta_semi_constant defaults",
                v->eta_semi_constant,
                NonIdealMHD{}.eta_semi_constant);
        } else {
            REQUIRE(false);
        }

        REQUIRE(!out_cfg.has_field_eta());
    }

} // namespace

NEW_TEST(Unittest, "shammodels/sph/config/mhd_nimhd_json_roundtrip", 1) {
    test_nimhd_json_roundtrip();
}

NEW_TEST(Unittest, "shammodels/sph/config/mhd_nimhd_json_backward_compat", 1) {
    test_nimhd_json_backward_compat();
}
