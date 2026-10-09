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

    using Config     = shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
    using EOS        = shammodels::basegodunov::EOSConfig<f64_3>;
    using Adiabatic  = EOS::Adiabatic;
    using Barotropic = EOS::Barotropic;

    Config json_roundtrip(const Config &in) {
        nlohmann::json j = in;
        return nlohmann::json::parse(j.dump(4)).get<Config>();
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::eos", 1) {

    { // default is adiabatic with gamma = 5/3
        Config cfg{};
        REQUIRE(std::holds_alternative<Adiabatic>(cfg.eos_config.config));
        REQUIRE_FLOAT_EQUAL(cfg.get_eos_gamma(), 5. / 3., 1e-15);
        cfg.check_config();
    }

    { // legacy setter maps onto the adiabatic EOS
        Config cfg{};
        cfg.set_eos_gamma(1.4);
        REQUIRE(std::holds_alternative<Adiabatic>(cfg.eos_config.config));
        REQUIRE_EQUAL(cfg.get_eos_gamma(), 1.4);
        REQUIRE_EQUAL(std::get<Adiabatic>(cfg.eos_config.config).get_spec().gamma(), 1.4);
    }

    { // adiabatic json roundtrip
        Config in{};
        in.eos_config.set_adiabatic(1.42);
        Config out = json_roundtrip(in);
        REQUIRE(std::holds_alternative<Adiabatic>(out.eos_config.config));
        REQUIRE_EQUAL(out.get_eos_gamma(), 1.42);
    }

    { // barotropic json roundtrip
        Config in{};
        in.eos_config.set_barotropic(1e-13, 0.2, 1.4);
        Config out          = json_roundtrip(in);
        const Barotropic *b = std::get_if<Barotropic>(&out.eos_config.config);
        REQUIRE(b != nullptr);
        if (b != nullptr) {
            REQUIRE_EQUAL(b->rho_crit, 1e-13);
            REQUIRE_EQUAL(b->cs0, 0.2);
            REQUIRE_EQUAL(b->gamma, 1.4);
        }
    }

    { // configs dumped before eos_config existed only store eos_gamma
        nlohmann::json j = Config{};
        j.erase("eos_config");
        j["eos_gamma"] = 1.3;
        Config out     = j.get<Config>();
        REQUIRE(std::holds_alternative<Adiabatic>(out.eos_config.config));
        REQUIRE_EQUAL(out.get_eos_gamma(), 1.3);
    }

    { // the solver does not support the barotropic EOS yet
        Config cfg{};
        cfg.eos_config.set_barotropic(1e-13, 0.2, 1.4);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
        REQUIRE_EXCEPTION_THROW(cfg.get_eos_gamma(), std::invalid_argument);
    }

    { // gamma <= 1 is rejected
        Config cfg{};
        cfg.set_eos_gamma(1.0);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }
}
