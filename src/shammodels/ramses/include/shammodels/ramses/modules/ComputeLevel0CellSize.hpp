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
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
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

    template<class Tvec, class TgridVec>
    class ComputeLevel0CellSize : public shamrock::solvergraph::INode {
        public:
        ComputeLevel0CellSize() {}

        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &patch_boxes;
            const shamrock::solvergraph::IPatchDataLayerRefs &refs;
            const shamrock::solvergraph::IFieldRefs<TgridVec> &spans_block_min;
            const shamrock::solvergraph::IFieldRefs<TgridVec> &spans_block_max;
            shamrock::solvergraph::ScalarsEdge<TgridVec> &level0_size;
        };

        void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> refs,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<TgridVec>> spans_block_min,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<TgridVec>> spans_block_max,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<TgridVec>> level0_size) {
            __internal_set_ro_edges({patch_boxes, refs, spans_block_min, spans_block_max});
            __internal_set_rw_edges({level0_size});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::IPatchDataLayerRefs>(1),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<TgridVec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<TgridVec>>(3),
                get_rw_edge<shamrock::solvergraph::ScalarsEdge<TgridVec>>(0)};
        }

        void _impl_evaluate_internal() {
            auto edges               = get_edges();
            edges.level0_size.values = edges.refs.get_const_refs().template map<TgridVec>(
                [&](u64 id_patch, const shamrock::patch::PatchDataLayer &pdat) {
                    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                    sham::DeviceBuffer<TgridVec> block_size_buf(pdat.get_obj_cnt(), dev_sched);

                    sham::EventList depends_list;
                    auto block_min_acc
                        = edges.spans_block_min.get_field(id_patch).get_buf().get_read_access(
                            depends_list);
                    auto block_max_acc
                        = edges.spans_block_max.get_field(id_patch).get_buf().get_read_access(
                            depends_list);
                    auto block_size_acc = block_size_buf.get_write_access(depends_list);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(
                            sycl::range<1>(pdat.get_obj_cnt()), [=](sycl::item<1> gid) {
                                u32 id             = gid.get_linear_id();
                                block_size_acc[id] = block_max_acc[id] - block_min_acc[id];
                            });
                    });
                    edges.spans_block_min.get_field(id_patch).get_buf().complete_event_state(e);
                    edges.spans_block_max.get_field(id_patch).get_buf().complete_event_state(e);
                    block_size_buf.complete_event_state(e);

                    auto patch_max = shamalgs::primitives::max(
                        dev_sched, block_size_buf, 0, pdat.get_obj_cnt());
                    return patch_max;
                });
        }

        inline virtual std::string _impl_get_label() const { return "ComputeLevel0CellSize"; };

        virtual std::string _impl_get_tex() const { return "ComputeLevel0CellSize"; };
    };

} // namespace shammodels::basegodunov::modules
