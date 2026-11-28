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
 * @file AddExtForceField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node adding a per cell acceleration field to the external forces acceleration
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_field)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_ext)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Add a per cell acceleration field (e.g. the self-gravity acceleration) to the
     * external forces acceleration
     *
     * `sizes` holds the block counts per patch, each block holding `block_size` cells. Only these
     * cells are updated, so `spans_a_field` can hold more blocks (e.g. ghost blocks stored after
     * the patch blocks).
     */
    template<class Tvec>
    class AddExtForceField : public shamrock::solvergraph::INode {

        u32 block_size;

        public:
        AddExtForceField(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "AddExtForceField"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
