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
 * @file ExchangeGhostLayerDebugDotGraph.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solver graph node for exchanging ghost layer data between distributed processes
 *
 * This file defines the ExchangeGhostLayerDebugDotGraph class, which is a solver graph node
 * responsible for managing the communication of ghost layer data across distributed computational
 * domains in the Shamrock hydrodynamics framework.
 */

#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shamrock::solvergraph {

    class ExchangeGhostLayerDebugDotGraph : public shamrock::solvergraph::INode {
        shamalgs::collective::DDSCommCache cache;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        ExchangeGhostLayerDebugDotGraph() {}

        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<u64> &object_counts;
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u64>> object_counts,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer) {
            __internal_set_ro_edges({object_counts});
            __internal_set_rw_edges({ghost_layer});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<u64>>(0),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal() {
            auto edges        = get_edges();
            auto &ghost_layer = edges.ghost_layer;

            std::vector<u64_3> debug_infos;
            ghost_layer.patchdatas.for_each(
                [&](u64 sender, u64 receiver, shamrock::patch::PatchDataLayer &pdat) {
                    debug_infos.push_back(u64_3{sender, receiver, pdat.get_obj_cnt()});
                });
            std::vector<u64_3> collected;
            shamalgs::collective::vector_allgatherv(debug_infos, collected, MPI_COMM_WORLD);

            std::vector<u64_3> object_counts;
            edges.object_counts.values.for_each([&](u64 id, u64 count) {
                object_counts.push_back(u64_3{id, count, shamcomm::world_rank()});
            });
            std::vector<u64_3> collected_object_counts;
            shamalgs::collective::vector_allgatherv(
                object_counts, collected_object_counts, MPI_COMM_WORLD);

            auto compute_threshold = [&]() -> std::tuple<u64, u64, u64> {
                std::vector<u64> values;
                values.reserve(collected.size());

                for (const u64_3 &info : collected) {
                    values.push_back(info.z());
                }

                std::sort(values.begin(), values.end());

                auto percentile = [&](double p) -> u64 {
                    size_t idx = static_cast<size_t>(p * (values.size() - 1));
                    return values[idx];
                };

                u64 p50 = percentile(0.50);
                u64 p80 = percentile(0.80);
                u64 p95 = percentile(0.95);

                return {p50, p80, p95};
            };

            auto [p50, p80, p95] = compute_threshold();

            if (shamcomm::world_rank() == 0) {

                std::string log
                    = " --- ExchangeGhostLayerDebugDotGraph debug infos (comm sizes) --- \n";

                logger::raw_ln("p50: ", p50, "p80: ", p80, "p95: ", p95);

                log += R"graph(
                    graph [
                        overlap=false,
                        nodesep=3,
                        ranksep=5
                    ];
                )graph";

                u32 current_subgraph = 0;
                log += shambase::format("subgraph cluster_{0} {{\n", current_subgraph);

                for (u64_3 &info : collected_object_counts) {
                    if (info.z() != current_subgraph) {
                        log += "}\n";
                        current_subgraph = info.z();
                        log += shambase::format("subgraph cluster_{0} {{\n", current_subgraph);
                    }
                    log += shambase::format(
                        "p_{0} [label=\"Patch {0} N={1}\"];\n", info.x(), info.y());
                }
                log += "}\n";

                log += "\n";
                for (u64_3 &info : collected) {

                    const char *edge_color = "green";
                    if (info.z() >= p95) {
                        edge_color = "red";
                    } else if (info.z() >= p80) {
                        edge_color = "darkgoldenrod";
                    } else if (info.z() >= p50) {
                        edge_color = "blue";
                    }

                    log += shambase::format(
                        "p_{0} -> p_{1} [xlabel={2}, color={3}, fontcolor={3}];\n",
                        info.x(),
                        info.y(),
                        info.z(),
                        edge_color);
                }
                log += " --- ExchangeGhostLayerDebugDotGraph debug infos (comm sizes) --- \n";
                logger::raw_ln(log);
            }
        }

        inline virtual std::string _impl_get_label() const {
            return "ExchangeGhostLayerDebugDotGraph";
        };

        inline virtual std::string _impl_get_tex() const { return ""; };
    };
} // namespace shamrock::solvergraph
