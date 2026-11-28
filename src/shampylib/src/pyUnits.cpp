// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyUnits.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamrock/io/units_json.hpp"
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <shamunits/constant_table.hpp>
#include <shamunits/prefix_table.hpp>
#include <shamunits/unit_table.hpp>
#include <memory>

ON_PYTHON_INIT {
    auto &m = root_module;

    using UnitSystem = shamunits::UnitSystem<f64>;

    py::class_<UnitSystem>(m, "UnitSystem")
        .def(
            py::init([](f64 unit_time,
                        f64 unit_length,
                        f64 unit_mass,
                        f64 unit_current,
                        f64 unit_temperature,
                        f64 unit_qte,
                        f64 unit_lumint) {
                return std::make_unique<UnitSystem>(
                    unit_time,
                    unit_length,
                    unit_mass,
                    unit_current,
                    unit_temperature,
                    unit_qte,
                    unit_lumint);
            }),
            py::kw_only(),
            py::arg("unit_time")        = 1,
            py::arg("unit_length")      = 1,
            py::arg("unit_mass")        = 1,
            py::arg("unit_current")     = 1,
            py::arg("unit_temperature") = 1,
            py::arg("unit_qte")         = 1,
            py::arg("unit_lumint")      = 1)
        .def(
            "get",
            [](UnitSystem &self, std::string name, i32 power, std::string pref) {
                return self.get(
                    shamunits::prefix::from_name(pref) * shamunits::units::from_name(name), power);
            },
            // py::arg("self"),
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref")  = "None")
        .def(
            "to",
            [](UnitSystem &self, std::string name, i32 power, std::string pref) {
                return self.to(
                    shamunits::prefix::from_name(pref) * shamunits::units::from_name(name), power);
            },
            // py::arg("self"),
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref")  = "None"

            )
        .def(
            "to_json",
            [](UnitSystem &self) {
                auto json_loads  = py::module_::import("json").attr("loads");
                nlohmann::json j = self;
                return json_loads(j.dump());
            },
            "Converts the unit system to a json like dictionary")
        .def(
            "from_json",
            [](UnitSystem &self, const py::object &json_data) {
                auto json_dumps = py::module_::import("json").attr("dumps");
                std::string s   = json_dumps(json_data).cast<std::string>();
                self            = nlohmann::json::parse(s).get<UnitSystem>();
            },
            "Sets the unit system from a json like dictionary");

    auto constants_cls
        = py::class_<shamunits::Constants<f64>>(m, "Constants").def(py::init([](UnitSystem s) {
              return std::make_unique<shamunits::Constants<f64>>(s);
          }));

    // One method per entry of the constants registry, e.g. Constants(usys).G(power=2): a constant
    // added to shamunits/constant_table.hpp appears here with no binding edit.
    for (const shamunits::NamedUnit &e : shamunits::constants::registry) {
        constants_cls.def(
            std::string(e.long_name).c_str(),
            [u = e.unit](shamunits::Constants<f64> &cte, i32 power) {
                return cte.get(u, power);
            },
            py::arg("power") = 1);
    }
}
