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
 * @file SolverGraphSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declare a class to register and retrieve nodes and edges from a unique container.
 *
 */

#include "shambase/exception.hpp"
#include "shamrock/solvergraph/JsonSerializable.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"

namespace shamrock::solvergraph {

    class SolverGraphSerializable : public SolverGraph {
        public:
        SolverGraphSerializable()
            : SolverGraph(
                  SolverGraphConstraint{
                      .name       = "SolverGraphSerializable",
                      .node_check = [](const std::shared_ptr<INode> &node) -> bool {
                          return false; // there is no clean mechanism to serialize a node + its
                                        // connexions
                      },
                      .edge_check = [](const std::shared_ptr<IEdge> &edge) -> bool {
                          return bool(std::dynamic_pointer_cast<JsonSerializable>(edge));
                      }}) {}

        ~SolverGraphSerializable() = default;
    };

    /**
     * @brief Serialize a SolverGraphSerializable to JSON.
     *
     * Writes an `"edges"` object keyed by edge name. Each edge value includes the
     * polymorphic `"type"` discriminator and the fields produced by
     * @ref JsonSerializable::to_json.
     */
    inline void to_json(nlohmann::json &j, const SolverGraphSerializable &p) {
        nlohmann::json edges = nlohmann::json::object();

        for (const std::string &name : p.get_edge_names()) {
            auto serializable
                = std::dynamic_pointer_cast<JsonSerializable>(p.get_edge_ptr_base(name));
            if (!bool(serializable)) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "Edge '{}' is registered in SolverGraphSerializable but is not "
                    "JsonSerializable",
                    name));
            }

            nlohmann::json edge_j;
            serializable->to_json(edge_j);
            edge_j["type"] = serializable->type_name();
            edges[name]    = std::move(edge_j);
        }

        j = nlohmann::json{{"edges", std::move(edges)}};
    }

    /**
     * @brief Deserialize a SolverGraphSerializable from JSON.
     *
     * Expects an `"edges"` object. Each entry is reconstructed via
     * @ref JsonSerializable::from_json and registered under its key. Edge types must
     * already be registered in @ref JsonSerializable_registry.
     *
     * Population is all-or-nothing: edges are first registered into a temporary
     * graph, and `p` is only replaced after every edge has been deserialized
     * successfully. On any exception, `p` is left unchanged.
     */
    inline void from_json(const nlohmann::json &j, SolverGraphSerializable &p) {
        if (!j.is_object() || !j.contains("edges") || !j.at("edges").is_object()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "Invalid JSON for SolverGraphSerializable: expected an object with an 'edges' "
                "object");
        }

        SolverGraphSerializable tmp{};
        const auto &edges = j.at("edges");

        for (auto it = edges.begin(); it != edges.end(); ++it) {
            const std::string &name         = it.key();
            const nlohmann::json &edge_json = it.value();

            std::shared_ptr<JsonSerializable> serializable = JsonSerializable::from_json(edge_json);
            auto edge = std::dynamic_pointer_cast<IEdge>(serializable);
            if (!bool(edge)) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "Deserialized type for edge '{}' does not inherit from IEdge", name));
            }

            tmp.register_edge_ptr_base(name, std::move(edge));
        }

        p = std::move(tmp);
    }
} // namespace shamrock::solvergraph
