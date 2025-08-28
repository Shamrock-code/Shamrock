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
 * @file OrientedAMRGraphEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::basegodunov::solvergraph {

    template<class Tvec, class TgridVec>
    class OrientedAMRGraphEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        using OrientedAMRGraph = modules::OrientedAMRGraph<Tvec, TgridVec>;

        shambase::DistributedData<OrientedAMRGraph> graph;

        inline virtual void free_alloc() { graph = {}; }

        inline shambase::DistributedData<std::reference_wrapper<modules::AMRGraph>>
        get_refs_dir(modules::Direction dir) {
            return graph.template map<std::reference_wrapper<modules::AMRGraph>>(
                [&](u64 id,
                    OrientedAMRGraph &neigh_graph) -> std::reference_wrapper<modules::AMRGraph> {
                    return std::ref(shambase::get_check_ref(neigh_graph.graph_links[dir]));
                });
        }
        inline shambase::DistributedData<std::reference_wrapper<modules::AMRGraph>>
        get_refs_dir(modules::Direction dir) const {
            return graph.template map<std::reference_wrapper<modules::AMRGraph>>(
                [&](u64 id, const OrientedAMRGraph &neigh_graph)
                    -> std::reference_wrapper<modules::AMRGraph> {
                    return std::ref(shambase::get_check_ref(neigh_graph.graph_links[dir]));
                });
        }
    };

} // namespace shammodels::basegodunov::solvergraph
