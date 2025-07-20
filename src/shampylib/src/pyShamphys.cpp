// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamphys.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamphys/BlackHoles.hpp"
#include "shamphys/HydroSoundwave.hpp"
#include "shamphys/Planets.hpp"
#include "shamphys/SedovTaylor.hpp"
#include "shamphys/SodTube.hpp"
#include "shamphys/orbits.hpp"
#include <pybind11/complex.h>
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

    shamphys_module.def(
        "get_binary_pair",
        [](double m1, double m2, double a, double e, double nu, double G) {
            return shamphys::get_binary_pair(m1, m2, a, e, nu, G);
        },
        py::kw_only(),
        py::arg("m1"),
        py::arg("m2"),
        py::arg("a"),
        py::arg("e"),
        py::arg("nu"),
        py::arg("G"),
        R"pbdoc(
        Compute the positions and velocities of two objects in a binary system
        m1 : Mass of the first object
        m2 : Mass of the second object
        a : Semi-major axis of the system
        e : Eccentricity of the system
        nu : True anomaly of the system (in radians)
        G : Gravitational constant
    )pbdoc");

    shamphys_module.def(
        "get_binary_pair",
        [](double m1,
           double m2,
           double a,
           double e,
           double nu,
           shamunits::UnitSystem<double> usys) {
            return shamphys::get_binary_pair(m1, m2, a, e, nu, usys);
        },
        py::kw_only(),
        py::arg("m1"),
        py::arg("m2"),
        py::arg("a"),
        py::arg("e"),
        py::arg("nu"),
        py::arg("units"),
        R"pbdoc(
        Compute the positions and velocities of two objects in a binary system
        m1 : Mass of the first object
        m2 : Mass of the second object
        a : Semi-major axis of the system
        e : Eccentricity of the system
        nu : True anomaly of the system (in radians)
        units : unit system
    )pbdoc");

    shamphys_module.def(
        "rotate_point",
        [](f64_3 point, double roll, double pitch, double yaw) {
            return shamphys::rotate_point(point, roll, pitch, yaw);
        },
        py::kw_only(),
        py::arg("point"),
        py::arg("roll"),
        py::arg("pitch"),
        py::arg("yaw"),
        R"pbdoc(
        Rotate a 3D point using Euler angles.
        point : 3D point as a f64_3
        roll : Rotation about the X-axis (in radians)
        pitch : Rotation about the Y-axis (in radians)
        yaw : Rotation about the Z-axis (in radians)
    )pbdoc");

    shamphys_module.def(
        "get_binary_rotated",
        [](double m1,
           double m2,
           double a,
           double e,
           double nu,
           double G,
           double roll,
           double pitch,
           double yaw) {
            return shamphys::get_binary_rotated(m1, m2, a, e, nu, G, roll, pitch, yaw);
        },
        py::kw_only(),
        py::arg("m1"),
        py::arg("m2"),
        py::arg("a"),
        py::arg("e"),
        py::arg("nu"),
        py::arg("G"),
        py::arg("roll"),
        py::arg("pitch"),
        py::arg("yaw"),
        R"pbdoc(
        Rotate a binary orbit by Euler angles and return the positions and
        velocities of the two objects.
        m1 : Mass of the first object
        m2 : Mass of the second object
        a : Semi-major axis of the system
        e : Eccentricity of the system
        nu : True anomaly of the system (in radians)
        G : Gravitational constant
        roll : Rotation about the X-axis (in radians)
        pitch : Rotation about the Y-axis (in radians)
        yaw : Rotation about the Z-axis (in radians)
    )pbdoc");

    shamphys_module.def(
        "get_binary_rotated",
        [](double m1,
           double m2,
           double a,
           double e,
           double nu,
           const shamunits::UnitSystem<double> usys,
           double roll,
           double pitch,
           double yaw) {
            return shamphys::get_binary_rotated(m1, m2, a, e, nu, usys, roll, pitch, yaw);
        },
        py::kw_only(),
        py::arg("m1"),
        py::arg("m2"),
        py::arg("a"),
        py::arg("e"),
        py::arg("nu"),
        py::arg("units"),
        py::arg("roll"),
        py::arg("pitch"),
        py::arg("yaw"));

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.HydroSoundwave");
    py::class_<shamphys::HydroSoundwave>(shamphys_module, "HydroSoundwave")
        .def(
            py::init([](f64 cs, f64 k, std::complex<f64> rho_tilde, std::complex<f64> v_tilde) {
                return std::make_unique<shamphys::HydroSoundwave>(
                    shamphys::HydroSoundwave{cs, k, rho_tilde, v_tilde});
            }),
            py::kw_only(),
            py::arg("cs"),
            py::arg("k"),
            py::arg("rho_tilde"),
            py::arg("v_tilde"))
        .def("get_omega", &shamphys::HydroSoundwave::get_omega)
        .def("get_value", [](shamphys::HydroSoundwave &self, f64 t, f64 x) {
            auto ret = self.get_value(t, x);
            return std::pair<f64, f64>{ret.rho, ret.v};
        });

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.SodTube");
    py::class_<shamphys::SodTube>(shamphys_module, "SodTube")
        .def(
            py::init([](f64 gamma, f64 rho_1, f64 P_1, f64 rho_5, f64 P_5) {
                return std::make_unique<shamphys::SodTube>(
                    shamphys::SodTube{gamma, rho_1, P_1, rho_5, P_5});
            }),
            py::kw_only(),
            py::arg("gamma"),
            py::arg("rho_1"),
            py::arg("P_1"),
            py::arg("rho_5"),
            py::arg("P_5"))
        .def("get_value", [](shamphys::SodTube &self, f64 t, f64 x) {
            auto ret = self.get_value(t, x);
            return std::tuple<f64, f64, f64>{ret.rho, ret.vx, ret.P};
        });

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.SedovTaylor");
    py::class_<shamphys::SedovTaylor>(shamphys_module, "SedovTaylor")
        .def(py::init([]() {
            return std::make_unique<shamphys::SedovTaylor>(shamphys::SedovTaylor{});
        }))
        .def("get_value", [](shamphys::SedovTaylor &self, f64 x) {
            auto ret = self.get_value(x);
            return std::tuple<f64, f64, f64>{ret.rho, ret.vx, ret.P};
        });
}
