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
 * @file TreeEdge.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamtree/RadixTree.hpp"

namespace shammodels::basegodunov::solvergraph {
    template<class Umorton, class Tvec>
    class TreeEdge : public shamrock::solvergraph::IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        using RTree = RadixTree<Umorton, Tvec>;

        shambase::DistributedData<RTree> trees;

        inline auto extract_trees() { return std::move(trees); }

        inline virtual void free_alloc() { trees = {}; }
    };

} // namespace shammodels::basegodunov::solvergraph
