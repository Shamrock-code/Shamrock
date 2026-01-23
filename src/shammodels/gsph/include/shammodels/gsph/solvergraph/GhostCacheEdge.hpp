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

    /**
     * @brief SolverGraph edge wrapping the ghost interface cache
     *
     * This edge stores the cache map used for ghost particle communication.
     * The cache is built per timestep based on the current particle distribution
     * and freed after the force computation.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    class GhostCacheEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        using GhostHandle = GSPHGhostHandler<Tvec>;
        using CacheMap    = typename GhostHandle::CacheMap;

        std::optional<CacheMap> cache;

        /**
         * @brief Get the ghost cache
         * @throws std::runtime_error if cache is not set
         */
        CacheMap &get() {
            if (!cache.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostCache not set");
            }
            return cache.value();
        }

        /**
         * @brief Get the ghost cache (const)
         * @throws std::runtime_error if cache is not set
         */
        const CacheMap &get() const {
            if (!cache.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostCache not set");
            }
            return cache.value();
        }

        /**
         * @brief Check if the cache is set
         */
        bool has_value() const { return cache.has_value(); }

        /**
         * @brief Set the ghost cache
         * @param c The cache map to store
         */
        void set(CacheMap &&c) { cache = std::move(c); }

        /**
         * @brief Free the allocated cache
         */
        inline virtual void free_alloc() override { cache.reset(); }
    };

} // namespace shammodels::gsph::solvergraph
