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
 * @file PyScriptHandle.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief utility to use python script withing tests

 * Exemple :
 * \code{.cpp}
 * TestStart(Unittest, "shamtest/PyScriptHandle(plot)", shamtestpyscriptplot, 1) {
 *
 *     std::vector<f64> x = {0, 1, 2, 4, 5};
 *     std::vector<f64> y = {1, 2, 4, 6, 1};
 *
 *     PyScriptHandle hdnl{};
 *
 *     hdnl.data()["x"] = x;
 *     hdnl.data()["y"] = y;
 *
 *     hdnl.exec(R"(
 *         import matplotlib.pyplot as plt
 *         plt.plot(x,y)
 *         plt.savefig("tests/figures/test.pdf")
 *     )");
 * }
 * \endcode
 * @date 2023-01-04
 */

#include "shambindings/pytypealias.hpp"
#include "shamsys/legacy/log.hpp"
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

/**
 * @brief Class allowing use of python scripts within a test case
 *
 * Exemple :
 * \code{.cpp}
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
 * \endcode
 */
struct PyScriptHandle {

    /**
     * @brief local python variables
     */
    pybind11::dict locals;

    /**
     * @brief Construct `PyScriptHandle` and create an instance of \ref PyScriptHandle.locals
     */
    PyScriptHandle() { std::make_unique<pybind11::dict>(); }

    /**
     * @brief return reference to the locals
     *
     * @return pybind11::dict& reference to \ref PyScriptHandle.locals
     */
    pybind11::dict &data() { return locals; }

    /**
     * @brief register an array in a local py variable
     *
     * @tparam T type of the array (has to be registered to python)
     * @param name name of the python variable
     * @param arr the array to pass
     */
    template<class T>
    inline void register_array(std::string name, std::vector<T> &arr) {
        data()[name.c_str()] = arr;
    }

    /**
     * @brief register a value in a local py variable
     *
     * @tparam T type of the value (has to be registered to python)
     * @param name name of the python variable
     * @param val the value to pass
     */
    template<class T>
    inline void register_value(std::string name, T val) {
        data()[name.c_str()] = val;
    }

    /**
     * @brief execute the given script
     *
     * @tparam N length of the source chars
     * @param expr the source to be executed
     * @param global the global py variable
     * @return true the script executed succesfully
     * @return false  the script executed with an error
     */
    template<size_t N>
    inline bool exec(const char (&expr)[N], pybind11::object &&global = pybind11::globals()) {
        try {
            py::exec(expr, std::forward<pybind11::object>(global), locals);
        } catch (const std::exception &e) {
            logger::warn_ln("PyScriptHandle", "the script throwed an error :", e.what());
            return false;
        }
        return true;
    }
};
