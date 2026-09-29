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
 * @file ComputePatchTreeMaxField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solvergraph node reducing a per patch value to its max over the serial patch tree nodes
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/PatchtreeFieldEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::SerialPatchTreeRefEdge<Tvec>, patch_tree)                          \
    X_RO(shamrock::solvergraph::ScalarsEdge<Tscal>, patch_values)                                  \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::PatchtreeFieldEdge<Tscal>, patchtree_max)

namespace shammodels::sph::modules {

    /**
     * @brief Compute the max of a per patch value over every node of the serial patch tree
     *
     * `patch_values` must hold a value for every patch of the tree (i.e. every global patch).
     *
     * @tparam Tvec position vector type of the serial patch tree
     */
    template<class Tvec>
    class ComputePatchTreeMaxField : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        ComputePatchTreeMaxField() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputePatchTreeMaxField"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
