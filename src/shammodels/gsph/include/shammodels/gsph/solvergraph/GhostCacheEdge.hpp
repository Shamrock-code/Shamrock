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
 * @file GhostCacheEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SolverGraph edge for GSPH ghost cache
 */

#include "shambase/memory.hpp"
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <optional>

namespace shammodels::gsph::solvergraph {

    /// SolverGraph edge for ghost interface cache
    template<class Tvec>
    class GhostCacheEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        using GhostHandle = GSPHGhostHandler<Tvec>;
        using CacheMap    = typename GhostHandle::CacheMap;

        std::optional<CacheMap> cache;

        CacheMap &get() {
            if (!cache.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostCache not set");
            }
            return cache.value();
        }

        const CacheMap &get() const {
            if (!cache.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostCache not set");
            }
            return cache.value();
        }

        bool has_value() const { return cache.has_value(); }
        void set(CacheMap &&c) { cache = std::move(c); }
        inline virtual void free_alloc() override { cache.reset(); }
    };

} // namespace shammodels::gsph::solvergraph
