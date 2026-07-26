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
 * @file PatchDataSetup.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Proxy for patch field get/set during Python IC / field setup.
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <functional>
#include <string>

namespace py = pybind11;

namespace shamrock {

    /**
     * @brief Thin proxy over named field getters/setters as float64 numpy arrays.
     *
     * Does not own PatchData. Real fields and virtual (computed) fields share the
     * same get/set path via registered lambdas.
     */
    class PatchDataSetup {
        std::unordered_map<std::string, std::function<py::array_t<f64>()>> getters;
        std::unordered_map<std::string, std::function<void(py::array_t<f64>)>> setters;

        public:
        void register_getter(std::string name, std::function<py::array_t<f64>()> fn) {
            getters[std::move(name)] = std::move(fn);
        }

        void register_setter(std::string name, std::function<void(py::array_t<f64>)> fn) {
            setters[std::move(name)] = std::move(fn);
        }

        py::array_t<f64> get(const std::string &name) const {
            auto it = getters.find(name);
            if (it == getters.end()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "PatchDataSetup: no getter registered for field \"{}\"", name));
            }
            return it->second();
        }

        void set(const std::string &name, py::array_t<f64> value) const {
            auto it = setters.find(name);
            if (it == setters.end()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "PatchDataSetup: no setter registered for field \"{}\"", name));
            }
            it->second(std::move(value));
        }
    };

} // namespace shamrock
