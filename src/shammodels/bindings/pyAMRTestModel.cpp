// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRTestModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */
 

#include "shambindings/pybindaliases.hpp"

#include "shambindings/pybind11_stl.hpp"
#include <pybind11/numpy.h>

#include <memory>

#include "shammodels/amr/AMROverheadtest.hpp"

Register_pymod(pyamrtestmode){


    shamcomm::logs::debug_ln("[Py]","registering shamrock.AMRTestModel");
    py::class_<AMRTestModel>(m, "AMRTestModel")
        .def(py::init([](AMRTestModel::Grid & grd) {
                return std::make_unique<AMRTestModel>(grd);
            }))
        .def("refine",
        [](AMRTestModel& obj){
                obj.refine();
        }).def("derefine",
        [](AMRTestModel& obj){
                obj.derefine();
        }).def("step",
        [](AMRTestModel& obj){
                obj.step();
        });





}
