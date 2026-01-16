// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shamphys/collapse.hpp"
#include "shamphys/eos.hpp"
#include "shamphys/fmm/GreenFuncGravCartesian.hpp"
#include "shamphys/fmm/contract_grav_moment.hpp"
#include "shamphys/fmm/grav_moment_offset.hpp"
#include "shamphys/fmm/grav_moments.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
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

    shamphys_module.def(
        "free_fall_time",
        [](f64 rho, f64 G) {
            return shamphys::free_fall_time(rho, G);
        },
        py::arg("rho"),
        py::arg("G"),
        R"pbdoc(
        Compute the free fall time
        rho : Density
        G : Gravitational constant
    )pbdoc");

    shamphys_module.def(
        "free_fall_time",
        [](f64 rho, const shamunits::UnitSystem<f64> usys) {
            return shamphys::free_fall_time(rho, usys);
        },
        py::arg("rho"),
        py::arg("units"),
        R"pbdoc(
        Compute the free fall time
        rho : Density
        units : unit system
    )pbdoc");

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

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.green_func_grav_cartesian");
    shamphys_module.def(
        "green_func_grav_cartesian_0_5", &shamphys::green_func_grav_cartesian<f64, 0, 5>);
    shamphys_module.def(
        "green_func_grav_cartesian_0_4", &shamphys::green_func_grav_cartesian<f64, 0, 4>);
    shamphys_module.def(
        "green_func_grav_cartesian_0_3", &shamphys::green_func_grav_cartesian<f64, 0, 3>);
    shamphys_module.def(
        "green_func_grav_cartesian_0_2", &shamphys::green_func_grav_cartesian<f64, 0, 2>);
    shamphys_module.def(
        "green_func_grav_cartesian_0_1", &shamphys::green_func_grav_cartesian<f64, 0, 1>);
    shamphys_module.def(
        "green_func_grav_cartesian_0_0", &shamphys::green_func_grav_cartesian<f64, 0, 0>);
    shamphys_module.def(
        "green_func_grav_cartesian_1_5", &shamphys::green_func_grav_cartesian<f64, 1, 5>);
    shamphys_module.def(
        "green_func_grav_cartesian_1_4", &shamphys::green_func_grav_cartesian<f64, 1, 4>);
    shamphys_module.def(
        "green_func_grav_cartesian_1_3", &shamphys::green_func_grav_cartesian<f64, 1, 3>);
    shamphys_module.def(
        "green_func_grav_cartesian_1_2", &shamphys::green_func_grav_cartesian<f64, 1, 2>);
    shamphys_module.def(
        "green_func_grav_cartesian_1_1", &shamphys::green_func_grav_cartesian<f64, 1, 1>);

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.grav_moments");
    shamphys_module.def("get_M_mat_5", &shamphys::get_M_mat<f64, 0, 5>);
    shamphys_module.def("get_M_mat_4", &shamphys::get_M_mat<f64, 0, 4>);
    shamphys_module.def("get_M_mat_3", &shamphys::get_M_mat<f64, 0, 3>);
    shamphys_module.def("get_M_mat_2", &shamphys::get_M_mat<f64, 0, 2>);
    shamphys_module.def("get_M_mat_1", &shamphys::get_M_mat<f64, 0, 1>);
    shamphys_module.def("get_M_mat_0", &shamphys::get_M_mat<f64, 0, 0>);
    shamphys_module.def("get_dM_mat_5", &shamphys::get_dM_mat<f64, 4>);
    shamphys_module.def("get_dM_mat_4", &shamphys::get_dM_mat<f64, 3>);
    shamphys_module.def("get_dM_mat_3", &shamphys::get_dM_mat<f64, 2>);
    shamphys_module.def("get_dM_mat_2", &shamphys::get_dM_mat<f64, 1>);
    shamphys_module.def("get_dM_mat_1", &shamphys::get_dM_mat<f64, 0>);

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.contract_grav_moment_to_force");
    shamphys_module.def(
        "contract_grav_moment_to_force_5", &shamphys::contract_grav_moment_to_force<f64, 5>);
    shamphys_module.def(
        "contract_grav_moment_to_force_4", &shamphys::contract_grav_moment_to_force<f64, 4>);
    shamphys_module.def(
        "contract_grav_moment_to_force_3", &shamphys::contract_grav_moment_to_force<f64, 3>);
    shamphys_module.def(
        "contract_grav_moment_to_force_2", &shamphys::contract_grav_moment_to_force<f64, 2>);
    shamphys_module.def(
        "contract_grav_moment_to_force_1", &shamphys::contract_grav_moment_to_force<f64, 1>);

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.offset_multipole");
    shamphys_module.def(
        "offset_multipole_5",
        &shamphys::offset_multipole<f64, 0, 5>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_multipole_4",
        &shamphys::offset_multipole<f64, 0, 4>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_multipole_3",
        &shamphys::offset_multipole<f64, 0, 3>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_multipole_2",
        &shamphys::offset_multipole<f64, 0, 2>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_multipole_1",
        &shamphys::offset_multipole<f64, 0, 1>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_multipole_0",
        &shamphys::offset_multipole<f64, 0, 0>,
        py::arg("Q"),
        py::arg("from"),
        py::arg("to"));

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.phys.offset_dM_mat");
    shamphys_module.def(
        "offset_dM_mat_5",
        &shamphys::offset_dM_mat<f64, 1, 5>,
        py::arg("dM"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_dM_mat_4",
        &shamphys::offset_dM_mat<f64, 1, 4>,
        py::arg("dM"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_dM_mat_3",
        &shamphys::offset_dM_mat<f64, 1, 3>,
        py::arg("dM"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_dM_mat_2",
        &shamphys::offset_dM_mat<f64, 1, 2>,
        py::arg("dM"),
        py::arg("from"),
        py::arg("to"));
    shamphys_module.def(
        "offset_dM_mat_1",
        &shamphys::offset_dM_mat<f64, 1, 1>,
        py::arg("dM"),
        py::arg("from"),
        py::arg("to"));

    // EOS

    py::module eos_module = shamphys_module.def_submodule("eos");

    eos_module.def(
        "eos_Machida06",
        [](f64 cs, f64 rho, f64 rho_c1, f64 rho_c2, f64 rho_c3, f64 mu, f64 mh, f64 kb) {
            auto P   = shamphys::EOS_Machida06<f64>::pressure(cs, rho, rho_c1, rho_c2, rho_c3);
            auto _cs = shamphys::EOS_Machida06<f64>::soundspeed(P, rho, rho_c1, rho_c2, rho_c3);
            auto T   = shamphys::EOS_Machida06<f64>::temperature(P, rho, mu, mh, kb);
            return std::tuple<f64, f64, f64>{P, _cs, T};
        },
        py::kw_only(),
        py::arg("cs"),
        py::arg("rho"),
        py::arg("rho_c1"),
        py::arg("rho_c2"),
        py::arg("rho_c3"),
        py::arg("mu"),
        py::arg("mh"),
        py::arg("kb"));

    eos_module.def(
        "eos_Fermi",
        [](f64 mu_e, f64 rho) {
            auto P_cs = shamphys::EOS_Fermi<f64>::pressure_and_soundspeed(mu_e, rho);
            return std::tuple<f64, f64>{P_cs.pressure, P_cs.soundspeed};
        },
        py::kw_only(),
        py::arg("mu_e"),
        py::arg("rho"));
}
