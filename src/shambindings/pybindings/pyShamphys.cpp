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
#include "shambindings/pybindaliases.hpp"
#include "shamphys/BlackHoles.hpp"
#include "shamphys/Planets.hpp"

#include "shambindings/pybind11_stl.hpp"

Register_pymod(shamphyslibinit){

    py::module shamphys_module = m.def_submodule("phys", "Physics Library");

    // Planets.hpp

    shamphys_module.def("hill_radius",&shamphys::hill_radius<f64>, 
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

    shamphys_module.def("keplerian_speed",[](f64 G, f64 M, f64 R){
        return shamphys::keplerian_speed(G,M,R);
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

    shamphys_module.def("keplerian_speed",[](f64 M, f64 R, shamunits::UnitSystem<f64> usys){
        return shamphys::keplerian_speed(M,R,usys);
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

    shamphys_module.def("schwarzschild_radius",[](f64 M, f64 G, f64 c){
        return shamphys::schwarzschild_radius(M,G,c);
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

    shamphys_module.def("schwarzschild_radius",[](f64 M, shamunits::UnitSystem<f64> usys){
        return shamphys::schwarzschild_radius(M,usys);
    }, 
        py::kw_only(),
        py::arg("M"),
        py::arg("units"),
    R"pbdoc(
        Compute the schwarzschild radius
        M : Black hole mass
        units : unit system
    )pbdoc");
}