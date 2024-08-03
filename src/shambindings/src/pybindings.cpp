// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pybindings.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambindings/pybindaliases.hpp"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

/// With pybind we print using python out stream
void py_func_printer_normal(std::string s){
    using namespace pybind11;
    py::print(s,"end"_a="");
}

/// With pybind we print using python out stream
void py_func_printer_ln(std::string s){
    using namespace pybind11;
    py::print(s);
}

/// Python print performs already a flush so we need nothing here
void py_func_flush_func(){}

/// Statically initialized python module init function
std::vector<fct_sig> static_init_shamrock_pybind = {};

/// Add a python module init function to the init list
void register_pybind_init_func(fct_sig fct){
    static_init_shamrock_pybind.push_back(std::move(fct));
}

namespace shambindings {

    void init(py::module & m){
        #ifdef SHAMROCK_LIB_BUILD
        shambase::change_printer(&py_func_printer_normal, &py_func_printer_ln, &py_func_flush_func);
        #endif

        for(auto fct : static_init_shamrock_pybind){
            fct(m);
        }
    }

}

/// Call bindings init for the shamrock python module
SHAMROCK_PY_MODULE(shamrock,m){
    shambindings::init(m);
}