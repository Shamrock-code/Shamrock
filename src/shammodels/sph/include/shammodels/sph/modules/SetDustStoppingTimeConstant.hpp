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
 * @file SetDustStoppingTimeConstant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, t_j_0)                             \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SetDustStoppingTimeConstant : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;
        std::unique_ptr<sham::DeviceBuffer<Tscal>> t_j_0;

        public:
        SetDustStoppingTimeConstant(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts                   = edges.part_counts.indexes;
            const std::vector<Tscal> &inputs_tj = edges.t_j_0.value;
            SHAM_ASSERT(inputs_tj.size() == ndust);

            // ensure that the output edges are of size part_counts
            edges.t_j.ensure_sizes(part_counts);

            if (!t_j_0) {
                t_j_0 = std::make_unique<sham::DeviceBuffer<Tscal>>(
                    ndust, shamsys::instance::get_compute_scheduler_ptr());
            }
            t_j_0->resize(ndust);
            t_j_0->copy_from_stdvec(inputs_tj);

            auto &q = shamsys::instance::get_compute_scheduler().get_queue();

            part_counts.for_each([&](u64 id, u32 count) {
                // call the kernel for each patches with part_counts.get(id_patch) threads of patch
                // id_patch
                sham::kernel_call(
                    q,
                    sham::MultiRef{*t_j_0},
                    sham::MultiRef{edges.t_j.get_spans().get(id)},
                    count * ndust,
                    [ndust
                     = ndust](u32 thread_id, const Tscal *__restrict t_j_0, Tscal *__restrict t_j) {
                        u32 jdust      = thread_id % ndust;
                        t_j[thread_id] = t_j_0[jdust];
                    });
            });
        }

        inline virtual std::string _impl_get_label() const {
            return "SetDustStoppingTimeConstant";
        };

        inline virtual std::string _impl_get_tex() const {

            auto t_j_0       = get_ro_edge_base(0).get_tex_symbol();
            auto part_counts = get_ro_edge_base(1).get_tex_symbol();
            auto t_j         = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                SetDustStoppingTimeConstant

                \begin{align}
                {t_j}_{i,j} &= {t_j_0}_j \\
                i &\in [0,{part_counts}) \\
                j &\in [0,{ndust})
                \end{align}
            )tex";

            shambase::replace_all(tex, "{t_j_0}", t_j_0);
            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{ndust}", shambase::format("{}", ndust));
            shambase::replace_all(tex, "{t_j}", t_j);

            return tex;
        };
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
