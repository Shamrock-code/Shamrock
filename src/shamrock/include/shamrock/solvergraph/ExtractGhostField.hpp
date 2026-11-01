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
 * @file ExtractGhostField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief  This module allows to extract ghosts for a generic field such as density, velocity, etc
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, original_fields)                                    \
    X_RO(shamrock::solvergraph::DDSharedBuffers<u32>, idx_in_ghosts)                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::PatchDataFieldDDShared<T>, ghost_fields)

namespace shamrock::solvergraph {

    template<class T>
    class ExtractGhostField : public INode {

        public:
        ExtractGhostField() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ExtractGhostField"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shamrock::solvergraph

#undef NODE_EDGES
