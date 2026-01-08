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
 * @file FunctorNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Generic functor-based node for wrapping existing methods
 *
 * This node executes a user-provided std::function during evaluation.
 * Useful for wrapping existing solver methods as graph nodes without
 * creating separate classes for each operation.
 */

#include "shambase/stacktrace.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <functional>
#include <string>
#include <utility>

namespace shammodels::gsph::modules {

    /**
     * @brief A node that executes a captured functor during evaluation
     *
     * Unlike NodeSetEdge which modifies a single edge, FunctorNode simply
     * executes a stored function. The function can capture any references
     * it needs from the enclosing scope.
     *
     * @code{.cpp}
     * // Example: Wrap compute_eos as a node
     * auto compute_eos_node = FunctorNode(
     *     "compute_eos",
     *     "Compute EOS",
     *     [&]() { physics_mode->compute_eos(storage, config, scheduler); }
     * );
     * @endcode
     */
    class FunctorNode : public shamrock::solvergraph::INode {

        std::string label_;
        std::string description_;
        std::function<void()> functor_;

        public:
        /**
         * @brief Construct a FunctorNode
         *
         * @param label Short label for the node (used in DOT graphs)
         * @param description Longer description for documentation
         * @param functor The function to execute during evaluation
         */
        FunctorNode(std::string label, std::string description, std::function<void()> functor)
            : label_(std::move(label)), description_(std::move(description)),
              functor_(std::move(functor)) {}

        void _impl_evaluate_internal() override {
            StackEntry stack_loc{};
            functor_();
        }

        std::string _impl_get_label() const override { return label_; }

        std::string _impl_get_tex() const override { return description_; }
    };

} // namespace shammodels::gsph::modules
