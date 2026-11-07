// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/ramses/SolverConfig.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <variant>

namespace {

    using Config            = shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
    using EOS               = shammodels::basegodunov::EOSConfig<f64_3>;
    using Cooling           = shammodels::basegodunov::CoolingConfig<f64_3>;
    using Adiabatic         = EOS::Adiabatic;
    using NoCooling         = Cooling::None;
    using InstantBarotropic = Cooling::InstantBarotropic;

    Config json_roundtrip(const Config &in) {
        nlohmann::json j = in;
        return nlohmann::json::parse(j.dump(4)).get<Config>();
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::cooling", 1) {

    { // default is no cooling
        Config cfg{};
        REQUIRE(std::holds_alternative<NoCooling>(cfg.cooling_config.config));
        REQUIRE(!cfg.is_cooling_on());
        cfg.check_config();
    }

    { // no cooling json roundtrip
        Config out = json_roundtrip(Config{});
        REQUIRE(std::holds_alternative<NoCooling>(out.cooling_config.config));
    }

    { // instant barotropic json roundtrip, the EOS is kept as is
        Config in{};
        in.set_eos_gamma(1.001);
        in.cooling_config.set_instant_barotropic(1e-13, 0.2, 1.4);
        Config out                 = json_roundtrip(in);
        const InstantBarotropic *b = std::get_if<InstantBarotropic>(&out.cooling_config.config);
        REQUIRE(b != nullptr);
        if (b != nullptr) {
            REQUIRE_EQUAL(b->rho0, 1e-13);
            REQUIRE_EQUAL(b->cs0, 0.2);
            REQUIRE_EQUAL(b->gamma0, 1.4);
        }
        REQUIRE(std::holds_alternative<Adiabatic>(out.eos_config.config));
        REQUIRE_EQUAL(out.get_eos_gamma(), 1.001);
    }

    { // instant barotropic: the hydro step stays adiabatic, the barotrope uses gamma0
        Config cfg{};
        cfg.set_eos_gamma(1.001);
        cfg.cooling_config.set_instant_barotropic(1e-13, 0.2, 1.4);
        cfg.check_config();
        REQUIRE(cfg.is_cooling_on());
        REQUIRE(cfg.eos_config.is_hydro_adiabatic());
        REQUIRE_EQUAL(cfg.get_eos_gamma(), 1.001);

        const InstantBarotropic &b = std::get<InstantBarotropic>(cfg.cooling_config.config);
        REQUIRE_EQUAL(b.get_barotrope().gamma_adiab(), 1.4);
        // at rho0 the isothermal and adiabatic parts of the pressure are equal
        REQUIRE_FLOAT_EQUAL(
            b.get_barotrope().pressure(1e-13) / (2 * 0.2 * 0.2 * 1e-13), 1.0, 1e-14);
    }

    { // instant barotropic: invalid parameters are rejected
        Config cfg{};
        cfg.cooling_config.set_instant_barotropic(0.0, 0.2, 1.4);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
        cfg.cooling_config.set_instant_barotropic(1e-13, 0.0, 1.4);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);

        // the EOS gamma is still checked with the cooling on
        cfg.cooling_config.set_instant_barotropic(1e-13, 0.2, 1.4);
        cfg.set_eos_gamma(1.0);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }

    { // configs dumped with the legacy reset_barotropic EOS are converted
        nlohmann::json j = Config{};
        j.erase("cooling_config");
        j["eos_config"]
            = {{"type", "reset_barotropic"},
               {"rho_crit", 1e-13},
               {"cs0", 0.2},
               {"gamma_baro", 1.4},
               {"gamma", 1.001}};
        Config out = j.get<Config>();

        REQUIRE(std::holds_alternative<Adiabatic>(out.eos_config.config));
        REQUIRE_EQUAL(out.get_eos_gamma(), 1.001);

        const InstantBarotropic *b = std::get_if<InstantBarotropic>(&out.cooling_config.config);
        REQUIRE(b != nullptr);
        if (b != nullptr) {
            REQUIRE_EQUAL(b->rho0, 1e-13);
            REQUIRE_EQUAL(b->cs0, 0.2);
            REQUIRE_EQUAL(b->gamma0, 1.4);
        }
    }

    { // configs dumped before the cooling existed have no cooling
        nlohmann::json j = Config{};
        j.erase("cooling_config");
        Config out = j.get<Config>();
        REQUIRE(std::holds_alternative<NoCooling>(out.cooling_config.config));
    }

    { // invalid cooling type is rejected
        nlohmann::json j    = Config{};
        j["cooling_config"] = {{"type", "not_a_cooling"}};
        REQUIRE_EXCEPTION_THROW(j.get<Config>(), std::runtime_error);
    }
}
