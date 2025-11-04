// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include <pybind11/pybind11.h>

/// alias to pybind11 namespace
namespace py = pybind11;

/// function signature used to register python modules
using fct_sig = std::function<void(py::module &)>;

/// Register a python module init function to be ran on init
void register_pybind_init_func(fct_sig);

/// Utility struct to register python modules through static init
struct PyBindStaticInit {
    /// Constructor to register the python init function using static init
    inline explicit PyBindStaticInit(fct_sig t) { register_pybind_init_func(std::move(t)); }
};

/**
 * @brief Register a python module init function using static initialisation
 *
 * Usage (in any source files) :
 * @code{.cpp}
 *
 * Register_pymod(<python init module name>){
 *
 *    // You can define stuff in the python module object `m` like so :
 *    py::class_<ShamrockCtx>(m, "Context")
 *        .def(py::init<>())
 *
 * }
 * @endcode
 */
#define Register_pymod(placeholdername)                                                            \
    void pymod_##placeholdername(py::module &m);                                                   \
    void (*pymod_ptr_##placeholdername)(py::module & m) = pymod_##placeholdername;                 \
    PyBindStaticInit pymod_class_obj_##placeholdername(pymod_ptr_##placeholdername);               \
    void pymod_##placeholdername(py::module &m)
