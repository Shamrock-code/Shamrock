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
 * @file shamrock_json_to_py_json.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utilities to convert JSON objects to Python objects and vice versa.
 * TODO: try to convert directly without using string parsing
 */

#include "nlohmann/json.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"

namespace shammodels::common {

    template<class T>
    inline py::object to_py_json(const T &self) {
        auto json_loads  = py::module_::import("json").attr("loads");
        nlohmann::json j = self;
        return json_loads(j.dump());
    }

    template<class T>
    inline T from_py_json(py::object json_data) {
        auto json_dumps = py::module_::import("json").attr("dumps");
        std::string j   = json_dumps(json_data).cast<std::string>();
        return nlohmann::json::parse(j).get<T>();
    }

    template<class TConfig>
    inline void add_json_defs(py::class_<TConfig> &cls) {
        cls.def(
            "to_json",
            [](TConfig &self) {
                return shammodels::common::to_py_json(self);
            },
            "Converts the config to a json like dictionary");

        cls.def(
            "from_json",
            [](TConfig &self, py::object json_data) {
                self = shammodels::common::from_py_json<TConfig>(json_data);
            },
            "Converts a json like dictionary to a config");
    }
} // namespace shammodels::common
