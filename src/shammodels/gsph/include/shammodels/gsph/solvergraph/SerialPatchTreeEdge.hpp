// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SerialPatchTreeEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SolverGraph edge for SerialPatchTree
 */

#include "shambase/memory.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <optional>

namespace shammodels::gsph::solvergraph {

    /**
     * @brief SolverGraph edge wrapping the SerialPatchTree
     *
     * This edge stores the serial patch tree used for load balancing
     * and ghost particle interface detection. The tree is rebuilt
     * each timestep based on the current particle distribution.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    class SerialPatchTreeEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        std::optional<SerialPatchTree<Tvec>> tree;

        /**
         * @brief Get the serial patch tree
         * @throws std::runtime_error if tree is not set
         */
        SerialPatchTree<Tvec> &get() {
            if (!tree.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("SerialPatchTree not set");
            }
            return tree.value();
        }

        /**
         * @brief Get the serial patch tree (const)
         * @throws std::runtime_error if tree is not set
         */
        const SerialPatchTree<Tvec> &get() const {
            if (!tree.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("SerialPatchTree not set");
            }
            return tree.value();
        }

        /**
         * @brief Check if the tree is set
         */
        bool has_value() const { return tree.has_value(); }

        /**
         * @brief Set the serial patch tree
         * @param t The tree to store
         */
        void set(SerialPatchTree<Tvec> &&t) { tree = std::move(t); }

        /**
         * @brief Free the allocated tree
         */
        inline virtual void free_alloc() override { tree.reset(); }
    };

} // namespace shammodels::gsph::solvergraph
