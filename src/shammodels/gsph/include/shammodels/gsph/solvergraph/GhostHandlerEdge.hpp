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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief SolverGraph edge for GSPH ghost handler
 */

#include "shambase/memory.hpp"
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <optional>

namespace shammodels::gsph::solvergraph {

    /// SolverGraph edge for GSPH ghost handler
    template<class Tvec>
    class GhostHandlerEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        using GhostHandle = GSPHGhostHandler<Tvec>;

        std::optional<GhostHandle> handler;

        GhostHandle &get() {
            if (!handler.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostHandler not set");
            }
            return handler.value();
        }

        const GhostHandle &get() const {
            if (!handler.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("GhostHandler not set");
            }
            return handler.value();
        }

        bool has_value() const { return handler.has_value(); }

        void set(GhostHandle &&h) {
            handler.reset();
            handler.emplace(std::move(h));
        }

        /// Free the allocated handler
        inline virtual void free_alloc() override { handler.reset(); }
    };

} // namespace shammodels::gsph::solvergraph
