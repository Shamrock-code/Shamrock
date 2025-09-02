// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pybindings.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambindings/pybindaliases.hpp"
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <exception>
#include <memory>
#include <stdexcept>

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

void register_py_to_sham_print(py::module &m) {

    struct wrapper_cout {
        wrapper_cout()                     = default;
        wrapper_cout(const wrapper_cout &) = default;
        wrapper_cout(wrapper_cout &&)      = default;
    };

    struct wrapper_cerr {
        wrapper_cerr()                     = default;
        wrapper_cerr(const wrapper_cerr &) = default;
        wrapper_cerr(wrapper_cerr &&)      = default;
    };

    m.def("hook_stdout", [&]() {
        py::class_<wrapper_cout> my_stdout(m, "wrapper_cout");
        my_stdout.def(py::init<>());
        my_stdout.def_static("write", [](py::object buffer) {
            shambase::print(buffer.cast<std::string>());
        });
        my_stdout.def_static("flush", []() {
            shambase::flush();
        });

        py::class_<wrapper_cerr> my_stderr(m, "wrapper_cerr");
        my_stderr.def(py::init<>());
        my_stderr.def_static("write", [](py::object buffer) {
            shambase::print(buffer.cast<std::string>());
        });
        my_stderr.def_static("flush", []() {
            shambase::flush();
        });

        try {
            auto sys           = py::module::import("sys");
            sys.attr("stdout") = my_stdout();
            sys.attr("stderr") = my_stderr();
        } catch (std::exception &e) {
            shambase::throw_with_loc<std::runtime_error>(e.what());
        }
    });
}

namespace shambindings {

    enum { None = 0, Lib = 1, Embed = 2 } init_state = None;

    template<bool is_lib_mode>
    void init(py::module &m) {

        m.attr("__doc__") = R"doc(Python bindings for Shamrock)doc";

        if (is_lib_mode) {
            shambase::change_printer(
                &py_func_printer_normal, &py_func_printer_ln, &py_func_flush_func);
        }

        if (static_init_shamrock_pybind) {
            for (auto fct : *static_init_shamrock_pybind) {
                fct(m);
            }
        }

        if (is_lib_mode) {
            init_state = Lib;
        } else {
            init_state = Embed;
        }
    }

    void init_lib(py::module &m) { shambindings::init<true>(m); }

    void init_embed(py::module &m) {
        shambindings::init<false>(m);
        register_py_to_sham_print(m);
        m.attr("hook_stdout")();
    }

    void expect_init_lib(SourceLocation loc) {
        if (init_state != Lib) {
            shambase::throw_with_loc<std::runtime_error>(
                shambase::format(
                    "python bindings not initialized as lib mode, current mode = {}",
                    i32(init_state)),
                loc);
        }
    }

    void expect_init_embed(SourceLocation loc) {
        if (init_state != Embed) {
            shambase::throw_with_loc<std::runtime_error>(
                shambase::format(
                    "python bindings not initialized as embed mode, current mode = {}",
                    i32(init_state)),
                loc);
        }
    }

} // namespace shambindings
