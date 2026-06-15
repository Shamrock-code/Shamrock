// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pybindaliases.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Pybind11 include and definitions
 *
 * If we build shamrock executable we embed python in Shamrock
 * hence the include pybind11/embed.h.
 * If we build shamrock as python lib we import pybind11/pybind11.h.
 * Both options defines a similar syntax for the python module definition,
 * we can then wrap them conveniently in a single macro call.
 */

#include "shambase/call_lambda.hpp"
#include "shambase/unique_name_macro.hpp"
#include <pybind11/pybind11.h>

/// alias to pybind11 namespace
namespace py = pybind11;

/// function signature used to register python modules
using fct_sig = std::function<void(py::module &)>;

/// Register a python module init function to be ran on init
void register_pybind_init_func(fct_sig);

/**
 * @brief Internal helper that creates static symbols to register a Python init function via a
 * static initializer. It declares `funcname`, creates a `call_lambda` object that calls
 * `register_pybind_init_func(funcname)` at startup, and defines `funcname`.
 *
 * Here the objects/func are static in order to avoid conflicting name in linking. This is similar
 * to anonymous namespaces
 */
#define _internal_register_pybind_init(funcname, lambda_name, varname)                             \
    static void funcname(py::module &varname);                                                     \
    static shambase::call_lambda lambda_name([]() {                                                \
        register_pybind_init_func(funcname);                                                       \
    });                                                                                            \
    static void funcname(py::module &varname)

/**
 * @brief Register a Python module init function using static initialization
 *
 * Generates unique symbols automatically, making it convenient for one-shot initializations in
 * .cpp files.
 *
 * Usage (in a .cpp file) :
 * @code{.cpp}
 *
 * ON_PYTHON_INIT {
 *
 *    // Define things in the python module object `root_module` like so :
 *    root_module.def("hello", []() { return "Hello from SHAMROCK!"; });
 *
 * }
 * @endcode
 */
#define ON_PYTHON_INIT                                                                             \
    _internal_register_pybind_init(                                                                \
        __shamrock_unique_name(pybind_), __shamrock_unique_name(pybind_class_obj_), root_module)
