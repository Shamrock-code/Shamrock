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
 * @file ComputeLevel0CellSize.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shammath/AABB.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class ComputeLevel0CellSize : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        ComputeLevel0CellSize() {}

        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &patch_boxes;
            const shamrock::solvergraph::IPatchDataLayerRefs &refs;
            shamrock::solvergraph::ScalarsEdge<TgridVec> &level0_size;
        };

        void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> refs,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<TgridVec>> level0_size) {
            __internal_set_ro_edges({patch_boxes, refs});
            __internal_set_rw_edges({level0_size});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::IPatchDataLayerRefs>(1),
                get_rw_edge<shamrock::solvergraph::ScalarsEdge<TgridVec>>(0)};
        }

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
