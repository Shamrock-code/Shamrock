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
 * @file JsonSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/solvergraph/IEdge.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace shamrock::solvergraph {

    struct JsonSerializable {
        virtual ~JsonSerializable() {};

        virtual void to_json(nlohmann::json &j)         = 0;
        virtual void from_json(const nlohmann::json &j) = 0;

        virtual std::string type_name() = 0;
    };

    inline bool json_serializable_edge_constraint(
        const std::shared_ptr<shamrock::solvergraph::IEdge> &edge) {
        // check that the edge can be cross-casted to JsonSerializable
        return bool(std::dynamic_pointer_cast<JsonSerializable>(edge));
    };
} // namespace shamrock::solvergraph
