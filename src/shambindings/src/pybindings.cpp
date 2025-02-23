// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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
#include <memory>

// Before we used to redirect std::cout to python stdout but this creates deadlocks
// in adaptive cpp, hence the use of python printing functions

/// With pybind we print using python out stream
void py_func_printer_normal(std::string s) {
    using namespace pybind11;
    py::print(s, "end"_a = "");
}

/// With pybind we print using python out stream
void py_func_printer_ln(std::string s) {
    using namespace pybind11;
    py::print(s);
}

/// Python print performs already a flush so we need nothing here
void py_func_flush_func() {}

/**
 * @brief Statically initialized python module init function list
 * We use a unique pointer to ensure that the vector is not reinitialized
 * during programm init which would empty the function list otherwise
 *
 * This behavior was observed when building shamrock using object libraries
 * using the unique_ptr fixes it
 */
std::unique_ptr<std::vector<fct_sig>> static_init_shamrock_pybind;

/// Add a python module init function to the init list
void register_pybind_init_func(fct_sig fct) {
    if (!bool(static_init_shamrock_pybind)) {
        static_init_shamrock_pybind = std::make_unique<std::vector<fct_sig>>();
    }

    static_init_shamrock_pybind->push_back(std::move(fct));
}

namespace shambindings {

    void init(py::module &m) {
#ifdef SHAMROCK_LIB_BUILD
        shambase::change_printer(&py_func_printer_normal, &py_func_printer_ln, &py_func_flush_func);
#endif

        if (static_init_shamrock_pybind) {
            for (auto fct : *static_init_shamrock_pybind) {
                fct(m);
            }
        }
    }

} // namespace shambindings

/// Call bindings init for the shamrock python module
SHAMROCK_PY_MODULE(shamrock, m) {
    m.attr("__doc__") = R"doc(Python bindings for Shamrock)doc";
    shambindings::init(m);
}
