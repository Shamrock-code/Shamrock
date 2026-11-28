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
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <shamunits/constant_table.hpp>
#include <stdexcept>

namespace {

    using Config = shammodels::basegodunov::SolverConfig<f64_3, i64_3>;

    Config json_roundtrip(const Config &in) {
        nlohmann::json j = in;
        return nlohmann::json::parse(j.dump(4)).get<Config>();
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::gravity", 1) {

    { // defaults
        Config cfg{};
        REQUIRE_EQUAL(cfg.get_grav_tol(), 1e-6);
        REQUIRE_EQUAL(cfg.get_happy_bk_grav_tol(), 1e-6);
        REQUIRE_EQUAL(cfg.gravity_config.Niter_max, 100u);
    }

    { // 4 pi G uses the gravitational constant of the unit system, G = 1 with a mass unit of
      // 1 / G_SI kg
        f64 G_SI = shamunits::UnitSystem<f64>{}.get<shamunits::constants::G>();
        Config cfg{};
        cfg.set_units(shamunits::UnitSystem<f64>(1, 1, 1 / G_SI));
        REQUIRE_EQUAL(cfg.get_constant_G(), 1.);
        REQUIRE_FLOAT_EQUAL(cfg.get_constant_4piG(), 4 * shamunits::pi<f64>, 1e-15);
    }

    { // json roundtrip
        Config in{};
        in.gravity_config.gravity_mode = shammodels::basegodunov::PCG;
        in.gravity_config.tol          = 1e-9;
        in.gravity_config.tol_hp_bk    = 1e-12;
        in.gravity_config.Niter_max    = 1234;

        Config out = json_roundtrip(in);
        REQUIRE(out.gravity_config.gravity_mode == shammodels::basegodunov::PCG);
        REQUIRE_EQUAL(out.gravity_config.tol, 1e-9);
        REQUIRE_EQUAL(out.gravity_config.tol_hp_bk, 1e-12);
        REQUIRE_EQUAL(out.gravity_config.Niter_max, 1234u);
    }

    { // configs dumped before the self-gravity solver parameters existed still load
        Config in{};
        in.gravity_config.tol = 1e-7;

        nlohmann::json j = in;
        j.erase("gravity_tol_happy_breakdown");
        j.erase("gravity_Niter_max");

        Config out = j.get<Config>();
        REQUIRE_EQUAL(out.gravity_config.tol, 1e-7);
        REQUIRE_EQUAL(out.gravity_config.tol_hp_bk, 1e-6);
        REQUIRE_EQUAL(out.gravity_config.Niter_max, 100u);
    }
}

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::gravity_check_config", 1) {

    using namespace shammodels::basegodunov;

    // a valid self-gravity config, except for the tested option (the checks below are done
    // before the experimental feature check)
    auto make_cfg = []() {
        Config cfg{};
        cfg.gravity_config.gravity_mode = CG;
        cfg.ext_force_config.set_coupling(ExtForceCoupling::Trapezoidal);
        return cfg;
    };

    { // the midpoint coupling (default) is rejected
        Config cfg = make_cfg();
        cfg.ext_force_config.set_coupling(ExtForceCoupling::Midpoint);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }

    { // solvers that are not implemented
        Config cfg                      = make_cfg();
        cfg.gravity_config.gravity_mode = PCG;
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
        cfg.gravity_config.gravity_mode = MULTIGRID;
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }

    { // dust
        Config cfg                          = make_cfg();
        cfg.dust_config.dust_riemann_config = HB;
        cfg.dust_config.ndust               = 1;
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }

    { // drag
        Config cfg                         = make_cfg();
        cfg.drag_config.drag_solver_config = IRK1;
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }

    { // no self-gravity: the midpoint coupling is fine
        Config cfg{};
        cfg.check_config();
    }
}
