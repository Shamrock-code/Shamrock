// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambindings/pytypealias.hpp"
#include "shamsys/legacy/log.hpp"
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

/**
 * @brief Class allowing use of python scripts within a test case
 * Exemple : 
 * ~~~~~{.cpp}
 * std::vector<f64> x = {0, 1, 2, 4, 5};
 * std::vector<f64> y = {1, 2, 4, 6, 1};
 * 
 * PyScriptHandle hdnl{};
 * 
 * hdnl.data()["x"] = x;
 * hdnl.data()["y"] = y;
 * 
 * hdnl.exec(R"(
 *     print("startpy")
 *     import matplotlib.pyplot as plt
 *     plt.plot(x,y)
 *     plt.show()
 * )");
 * ~~~~~
 */
struct PyScriptHandle {
    pybind11::dict locals;

    PyScriptHandle() {
        std::make_unique<pybind11::dict>();
    }

    pybind11::dict &data() { return locals; }

    template<class T>
    inline void register_array(std::string name, std::vector<T> & arr){
        data()[name.c_str()] = arr;
    }

    template<class T>
    inline void register_value(std::string name, T val){
        data()[name.c_str()] = val;
    }

    template <size_t N>
    inline bool exec(const char (&expr)[N], pybind11::object &&global = pybind11::globals()) {
        try{
            py::exec(expr, std::forward<pybind11::object>(global), locals);
        } catch (const std::exception &e) {
            logger::warn_ln("PyScriptHandle", "the script throwed an error :",e.what());
            return false;
        }
        return true;
    }
};