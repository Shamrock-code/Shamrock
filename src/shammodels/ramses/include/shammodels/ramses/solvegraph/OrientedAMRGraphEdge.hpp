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
 * @file OrientedAMRGraphEdge.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"

namespace shammodels::basegodunov::solvergraph {

    template<class Tvec, class TgridVec>
    class OrientedAMRGraphEdge : public shamrock::solvergraph::IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        using OrientedAMRGraph = modules::OrientedAMRGraph<Tvec, TgridVec>;

        shambase::DistributedData<OrientedAMRGraph> graph;

        inline virtual void free_alloc() { graph = {}; }
    };

} // namespace shammodels::basegodunov::solvergraph
