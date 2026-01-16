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
#include <pybind11/cast.h>
#include <shamunits/Constants.hpp>
#include <shamunits/Names.hpp>
#include <shamunits/UnitSystem.hpp>
#include <memory>

Register_pymod(pyunits_init) {

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
                shamunits::UnitPrefix pref_ = shamunits::unit_prefix_from_name(pref);

                return self.runtime_get(pref_, shamunits::units::unit_from_name(name), power);
            },
            // py::arg("self"),
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref")  = "None")
        .def(
            "to",
            [](UnitSystem &self, std::string name, i32 power, std::string pref) {
                shamunits::UnitPrefix pref_ = shamunits::unit_prefix_from_name(pref);

                return self.runtime_to(pref_, shamunits::units::unit_from_name(name), power);
            },
            // py::arg("self"),
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref")  = "None"

        );

    py::class_<shamunits::Constants<f64>>(m, "Constants")
        .def(py::init([](UnitSystem s) {
            return std::make_unique<shamunits::Constants<f64>>(s);
        }))
////////////////// Xmacro
/// X macro to define the conversion constants bindings
#define X(st, conv)                                                                                \
    .def(                                                                                          \
        #st,                                                                                       \
        [](shamunits::Constants<f64> &cte, i32 power) {                                            \
            return sycl::pown(cte.st(), power);                                                    \
        },                                                                                         \
        py::arg("power") = 1)
        //////////////////
        UNITS_CONSTANTS
//////////////////
#undef X
        //////////////////

        ;
}
