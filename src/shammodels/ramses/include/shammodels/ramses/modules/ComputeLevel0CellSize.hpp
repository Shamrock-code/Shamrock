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
 * @file ComputeLevel0CellSize.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/primitives/reduction.hpp"
#include "shammath/AABB.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {
    template<class TgridVec>
    class ComputeLevel0CellSize : public shamrock::solvergraph::INode {
        public:
        ComputeLevel0CellSize() {}
#define NODE_ComputeLevel0CellSize_EDGES(X_RO, X_RW)                                               \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>, patch_boxes)                \
    X_RO(shamrock::solvergraph::IPatchDataLayerRefs, refs)                                         \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::ScalarsEdge<TgridVec>, level0_size)

        EXPAND_NODE_EDGES(NODE_ComputeLevel0CellSize_EDGES)
#undef NODE_ComputeLevel0CellSize_EDGES

        void _impl_evaluate_internal() {
            auto edges               = get_edges();
            edges.level0_size.values = edges.refs.get_const_refs().template map<TgridVec>(
                [&](u64 id_patch, const shamrock::patch::PatchDataLayer &pdat) {
                    shammath::AABB<TgridVec> patch_box = edges.patch_boxes.values.get(id_patch);
                    return patch_box.delt();
                });
        }

        inline virtual std::string _impl_get_label() const { return "ComputeLevel0CellSize"; };

        virtual std::string _impl_get_tex() const { return "ComputeLevel0CellSize"; };
    };

} // namespace shammodels::basegodunov::modules
