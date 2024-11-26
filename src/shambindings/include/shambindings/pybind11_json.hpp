// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pybind11_json.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>

namespace shambindings {

    pybind11::dict json_to_pydict(const nlohmann::json &j) {
        if (j.is_null()) {
            return pybind11::none();
        } else if (j.is_boolean()) {
            return pybind11::bool_(j.get<bool>());
        } else if (j.is_number_unsigned()) {
            return pybind11::int_(j.get<nlohmann::json::number_unsigned_t>());
        } else if (j.is_number_integer()) {
            return pybind11::int_(j.get<nlohmann::json::number_integer_t>());
        } else if (j.is_number_float()) {
            return pybind11::float_(j.get<double>());
        } else if (j.is_string()) {
            return pybind11::str(j.get<std::string>());
        } else if (j.is_array()) {
            pybind11::list obj(j.size());
            for (std::size_t i = 0; i < j.size(); i++) {
                obj[i] = json_to_pydict(j[i]);
            }
            return obj;
        } else if (j.is_object()) {
            pybind11::dict obj;
            for (nlohmann::json::const_iterator it = j.cbegin(); it != j.cend(); ++it) {
                obj[pybind11::str(it.key())] = json_to_pydict(it.value());
            }
            return obj;
        } else {
            shambase::throw_unimplemented(
                shambase::format("Unsupported json type : \n the object = {}", j.dump(4)));
        }
        return {};
    }

} // namespace shambindings
