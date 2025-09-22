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
 * @file FindGhostLayerCandidates.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"

namespace shammodels::basegodunov::modules {

    enum class GhostType { None, Periodic, Reflective };

    struct GhostLayerGenMode {
        GhostType ghost_type_x;
        GhostType ghost_type_y;
        GhostType ghost_type_z;
    };

    template<class TgridVec>
    shammath::paving_function_general_3d<TgridVec> get_paving(
        GhostLayerGenMode mode, shammath::AABB<TgridVec> sim_box) {

        TgridVec box_size   = sim_box.upper - sim_box.lower;
        TgridVec box_center = (sim_box.upper + sim_box.lower) / 2;

        SHAM_ASSERT(sim_box.is_volume_not_null());

        { // check that rebuildind the AABB from size and center gives the same AABB
            shammath::AABB<TgridVec> new_box
                = {box_center - box_size / 2, box_center + box_size / 2};
            if (new_box != sim_box) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Rebuilding AABB from size and center gives a different AABB");
            }
        }

        return shammath::paving_function_general_3d<TgridVec>{
            box_size,
            box_center,
            mode.ghost_type_x == GhostType::Periodic,
            mode.ghost_type_y == GhostType::Periodic,
            mode.ghost_type_z == GhostType::Periodic};
    }

    template<class Func>
    void for_each_paving_tile(GhostLayerGenMode mode, Func &&func) {

        // if the ghost type is none, we do not need to repeat as there is no ghost layer
        i32 repetition_x = mode.ghost_type_x != GhostType::None;
        i32 repetition_y = mode.ghost_type_y != GhostType::None;
        i32 repetition_z = mode.ghost_type_z != GhostType::None;

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {
                    func(xoff, yoff, zoff);
                }
            }
        }
    }

    struct GhostLayerCandidateInfos {
        i32 xoff;
        i32 yoff;
        i32 zoff;
    };

    template<class TgridVec>
    class FindGhostLayerCandidates : public shamrock::solvergraph::INode {

        GhostLayerGenMode mode;

        public:
        FindGhostLayerCandidates(GhostLayerGenMode mode) : mode(mode) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::IDataEdge<std::vector<u64>> &ids_to_check;
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>> &sim_box;
            const shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec> &patch_tree;
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &patch_boxes;
            // outputs
            shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>
                &ghost_layers_candidates;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<std::vector<u64>>> ids_to_check,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>> patch_tree,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
                ghost_layers_candidates) {
            __internal_set_ro_edges({ids_to_check, sim_box, patch_tree, patch_boxes});
            __internal_set_rw_edges({ghost_layers_candidates});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<std::vector<u64>>>(0),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(1),
                get_ro_edge<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(3),
                get_rw_edge<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "FindGhostLayerCandidates"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
