// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamphys.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shamcomm/logs.hpp"
#include "shamphys/SodTube.hpp"
#include <pybind11/complex.h>
#include "shambindings/pybindaliases.hpp"
#include "shamphys/BlackHoles.hpp"
#include "shamphys/HydroSoundwave.hpp"
#include "shamphys/Planets.hpp"
#include <complex>
#include <utility>

Register_pymod(shamphyslibinit) {

    py::module shamphys_module = m.def_submodule("phys", "Physics Library");

    // Planets.hpp

    shamphys_module.def(
        "hill_radius",
        &shamphys::hill_radius<f64>,
        py::kw_only(),
        py::arg("R"),
        py::arg("m"),
        py::arg("M"),
        R"pbdoc(
        Compute the hill radius of a planet
        R : Orbit radius
        m : planet mass
        M : Star mass
    )pbdoc");

    shamphys_module.def(
        "keplerian_speed",
        [](f64 G, f64 M, f64 R) {
            return shamphys::keplerian_speed(G, M, R);
        },
        py::kw_only(),
        py::arg("G"),
        py::arg("M"),
        py::arg("R"),
        R"pbdoc(
        Compute the keplerian orbit speed
        G : Gravitational constant
        M : planet mass
        R : orbit radius
    )pbdoc");

    shamphys_module.def(
        "keplerian_speed",
        [](f64 M, f64 R, shamunits::UnitSystem<f64> usys) {
            return shamphys::keplerian_speed(M, R, usys);
        },
        py::kw_only(),
        py::arg("M"),
        py::arg("R"),
        py::arg("units"),
        R"pbdoc(
        Compute the keplerian orbit speed
        M : planet mass
        R : orbit radius
        units : unit system
    )pbdoc");

    // BlackHole.hpp

    shamphys_module.def(
        "schwarzschild_radius",
        [](f64 M, f64 G, f64 c) {
            return shamphys::schwarzschild_radius(M, G, c);
        },
        py::kw_only(),
        py::arg("M"),
        py::arg("G"),
        py::arg("c"),
        R"pbdoc(
        Compute the schwarzschild radius 
        M : Black hole mass
        G : Gravitational constant
        c : Speed of light
    )pbdoc");

    shamphys_module.def(
        "schwarzschild_radius",
        [](f64 M, shamunits::UnitSystem<f64> usys) {
            return shamphys::schwarzschild_radius(M, usys);
        },
        py::kw_only(),
        py::arg("M"),
        py::arg("units"),
        R"pbdoc(
        Compute the schwarzschild radius
        M : Black hole mass
        units : unit system
    )pbdoc");

    shamcomm::logs::debug_ln("Bindings","registering shamrock.phys.HydroSoundwave");
    py::class_<shamphys::HydroSoundwave>(shamphys_module, "HydroSoundwave")
        .def(py::init([](f64 cs, f64 k, std::complex<f64> rho_tilde, std::complex<f64> v_tilde) {
            return std::make_unique<shamphys::HydroSoundwave>(
                shamphys::HydroSoundwave{cs, k, rho_tilde, v_tilde});
        }),
        py::kw_only(),
        py::arg("cs"),
        py::arg("k"),
        py::arg("rho_tilde"),
        py::arg("v_tilde"))
        .def("get_omega",&shamphys::HydroSoundwave::get_omega)
        .def("get_value",[](shamphys::HydroSoundwave& self,f64 t, f64 x){
            auto ret = self.get_value(t, x);
            return std::pair<f64,f64>{ret.rho, ret.v};
        });

    shamcomm::logs::debug_ln("Bindings","registering shamrock.phys.SodTube");
    py::class_<shamphys::SodTube>(shamphys_module, "SodTube")
        .def(py::init([](f64 gamma, f64 rho_1, f64 P_1, f64 rho_5, f64 P_5) {
            return std::make_unique<shamphys::SodTube>(
                shamphys::SodTube{gamma, rho_1, P_1, rho_5, P_5});
        }),
        py::kw_only(),
        py::arg("gamma"),
        py::arg("rho_1"),
        py::arg("P_1"),
        py::arg("rho_5"),
        py::arg("P_5")
        )
        .def("get_value",[](shamphys::SodTube& self,f64 t, f64 x){
            auto ret = self.get_value(t, x);
            return std::tuple<f64,f64,f64>{ret.rho, ret.vx, ret.P};
        });
}
