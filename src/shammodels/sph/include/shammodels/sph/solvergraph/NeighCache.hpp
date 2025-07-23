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
 * @file NeighCache.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::sph::solvergraph {

    class NeighCache : public shamrock::solvergraph::IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        shambase::DistributedData<shamrock::tree::ObjectCache> neigh_cache;

        shamrock::tree::ObjectCache &get_cache(u64 id) { return neigh_cache.get(id); }

        inline virtual void free_alloc() { neigh_cache = {}; }
    };

} // namespace shammodels::sph::solvergraph
