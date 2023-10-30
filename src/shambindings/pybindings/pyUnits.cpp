// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyUnits.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/Names.hpp>
#include <shamunits/UnitSystem.hpp>
#include <memory>
#include <pybind11/cast.h>
#include "shambackends/sycl.hpp"


Register_pymod(pyunits_init) {

    using UnitSystem = shamunits::UnitSystem<f64>;


    py::class_<UnitSystem>(m, "UnitSystem")
        .def(py::init([](f64 unit_time,
                         f64 unit_length,
                         f64 unit_mass,
                         f64 unit_current,
                         f64 unit_temperature,
                         f64 unit_qte,
                         f64 unit_lumint) {
            return std::make_unique<UnitSystem>(unit_time,
                                                unit_length,
                                                unit_mass,
                                                unit_current,
                                                unit_temperature,
                                                unit_qte,
                                                unit_lumint);
        }),
        py::kw_only(),
        py::arg("unit_time") = 1,
        py::arg("unit_length") = 1,
        py::arg("unit_mass") = 1,
        py::arg("unit_current") = 1,
        py::arg("unit_temperature") = 1,
        py::arg("unit_qte") = 1,
        py::arg("unit_lumint") = 1)
        .def("get",
            [](UnitSystem & self, std::string name, i32 power, std::string pref){

                shamunits::UnitPrefix pref_ =  shamunits::unit_prefix_from_name(pref);

                return self.runtime_get(
                        pref_, 
                        shamunits::units::unit_from_name(name), 
                        power
                    );

            }, 
            //py::arg("self"), 
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref") = "None"
        )
        .def("to",
            [](UnitSystem & self, std::string name, i32 power, std::string pref){
            
            
                shamunits::UnitPrefix pref_ =  shamunits::unit_prefix_from_name(pref);

                return self.runtime_to(
                    pref_, shamunits::units::unit_from_name(name), power);
            }, 
            //py::arg("self"), 
            py::arg("name"),
            py::arg("power") = 1,
            py::arg("pref") = "None"
            
        );



    py::class_<shamunits::Constants<f64>>(m, "Constants")
        .def(py::init([](UnitSystem s) {
            return std::make_unique<shamunits::Constants<f64>>(s);
        }))

        

        #define X(st) \
        .def( #st ,[](shamunits::Constants<f64> & cte, i32 power){\
            return sycl::pown(cte.st(),power);\
        },py::arg("power") = 1)

        X(delta_nu_cs)
        X(c)
        X(h)
        X(e)
        X(k)
        X(Na)
        X(Kcd)

        X(G)

        X(year)


        X(au)

        X(earth_mass)
        X(jupiter_mass)
        X(sol_mass)
        

        ;
}