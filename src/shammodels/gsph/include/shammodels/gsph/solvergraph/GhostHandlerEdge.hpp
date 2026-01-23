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
 * @file GhostHandlerEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SolverGraph edge for GSPH ghost handler
 */

#include "shambase/memory.hpp"
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <optional>

namespace shammodels::gsph::solvergraph {

    /**
     * @brief SolverGraph edge wrapping the GSPHGhostHandler
     *
     * This edge provides storage for the ghost handler used in GSPH
     * boundary condition processing. The handler is created per timestep
     * and freed after use.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    class GhostHandlerEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        using GhostHandle = GSPHGhostHandler<Tvec>;

        std::optional<GhostHandle> handler;

        /**
         * @brief Get the ghost handler
         * @throws std::runtime_error if handler is not set
         */
        GhostHandle &get() {
            if (!handler.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostHandler not set");
            }
            return handler.value();
        }

        /**
         * @brief Get the ghost handler (const)
         * @throws std::runtime_error if handler is not set
         */
        const GhostHandle &get() const {
            if (!handler.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostHandler not set");
            }
            return handler.value();
        }

        /**
         * @brief Check if the handler is set
         */
        bool has_value() const { return handler.has_value(); }

        /**
         * @brief Set the ghost handler
         * @param h The ghost handler to store
         */
        void set(GhostHandle &&h) {
            handler.reset();
            handler.emplace(std::move(h));
        }

        ///Free the allocated handler
        inline virtual void free_alloc() override { handler.reset(); }
    };

} // namespace shammodels::gsph::solvergraph
