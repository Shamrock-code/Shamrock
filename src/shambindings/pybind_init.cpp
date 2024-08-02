// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pybind_init.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/print.hpp"
#include "pybindaliases.hpp"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>


void py_func_printer_normal(std::string s){
    using namespace pybind11;
    py::print(s,"end"_a="");
}
void py_func_printer_ln(std::string s){
    using namespace pybind11;
    py::print(s);
}
void py_func_flush_func(){}

SHAMROCK_PY_MODULE(shamrock,m){

    #ifdef SHAMROCK_LIB_BUILD
    shambase::change_printer(&py_func_printer_normal, &py_func_printer_ln, &py_func_flush_func);
    #endif

    for(auto fct : static_init_shamrock_pybind){
        fct(m);
    }
}
