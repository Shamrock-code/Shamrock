// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/print.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/EOSConfig.hpp"
#include "shamphys/eos_config.hpp"
#include "shamtest/shamtest.hpp"

void test_serialize_adiabatic() {

    shammodels::EOSConfig<f64_3> in_config;

    in_config.set_adiabatic(1.42);

    nlohmann::json j = in_config;

    std::string s = j.dump(4);

    nlohmann::json jout = nlohmann::json::parse(s);

    shammodels::EOSConfig<f64_3> out_config = jout.template get<shammodels::EOSConfig<f64_3>>();

    using Config = shamphys::EOS_Config_Adiabatic<f64>;

    if (Config *out_eos_config = std::get_if<Config>(&out_config.config)) {
        if (Config *in_eos_config = std::get_if<Config>(&in_config.config)) {
            REQUIRE(*in_eos_config == *out_eos_config)
        } else {
            _Assert(false)
        }
    } else {
        _Assert(false)
    }
}

void test_serialize_locally_isothermal() {

    shammodels::EOSConfig<f64_3> in_config;

    in_config.set_locally_isothermal();

    nlohmann::json j = in_config;

    std::string s = j.dump(4);

    nlohmann::json jout = nlohmann::json::parse(s);

    shammodels::EOSConfig<f64_3> out_config = jout.template get<shammodels::EOSConfig<f64_3>>();

    using Config = shammodels::EOSConfig<f64_3>::LocallyIsothermal;

    if (Config *out_eos_config = std::get_if<Config>(&out_config.config)) {
        if (Config *in_eos_config = std::get_if<Config>(&in_config.config)) {
            _Assert(true)
        } else {
            _Assert(false)
        }
    } else {
        _Assert(false)
    }
}

void test_serialize_locally_isothermallp07() {

    shammodels::EOSConfig<f64_3> in_config;

    in_config.set_locally_isothermalLP07(1, 2, 3);

    nlohmann::json j = in_config;

    std::string s = j.dump(4);

    nlohmann::json jout = nlohmann::json::parse(s);

    shammodels::EOSConfig<f64_3> out_config = jout.template get<shammodels::EOSConfig<f64_3>>();

    using Config = shamphys::EOS_Config_LocallyIsothermal_LP07<f64>;

    if (Config *out_eos_config = std::get_if<Config>(&out_config.config)) {
        if (Config *in_eos_config = std::get_if<Config>(&in_config.config)) {
            REQUIRE(*in_eos_config == *out_eos_config)
        } else {
            _Assert(false)
        }
    } else {
        _Assert(false)
    }
}

TestStart(Unittest, "shammodels/EOSConfig::json", eosconfigserializejson, 1) {

    test_serialize_adiabatic();
    test_serialize_locally_isothermal();
    test_serialize_locally_isothermallp07();
    // TODO test others cases
}
