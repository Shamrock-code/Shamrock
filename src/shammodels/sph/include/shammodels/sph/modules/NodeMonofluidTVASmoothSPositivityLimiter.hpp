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
 * @file NodeMonofluidTVASmoothSPositivityLimiter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, Ttilde_sj)                                      \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, ds_j_dt)

namespace shammodels::sph::modules {

    template<class Tvec>
    class NodeMonofluidTVASmoothSPositivityLimiter : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        NodeMonofluidTVASmoothSPositivityLimiter(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            edges.ds_j_dt.ensure_sizes(part_counts);

            auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
                return count * ndust;
            });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.s_j.get_spans(), edges.Ttilde_sj.get_spans()},
                sham::DDMultiRef{edges.ds_j_dt.get_spans()},
                total_specie_count,
                [](u32 thread_id,
                   const Tscal *__restrict s_j,
                   const Tscal *__restrict Ttilde_sj,
                   Tscal *__restrict ds_j_dt) {
                    Tscal s_j_a       = s_j[thread_id];
                    Tscal Ttilde_sj_a = Ttilde_sj[thread_id];
                    Tscal ds_j_dt_a   = ds_j_dt[thread_id];

                    // if we dip in the negative range do not dip further
                    ds_j_dt_a *= (s_j_a < 0 && ds_j_dt_a < 0) ? 0 : 1;

                    // restore it slowly to 0
                    ds_j_dt_a += (s_j_a < 0) ? -s_j_a / (10 * Ttilde_sj_a) : 0;

                    ds_j_dt[thread_id] = ds_j_dt_a;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVASmoothSPositivityLimiter";
        };

        inline virtual std::string _impl_get_tex() const {

            auto part_counts = get_ro_edge_base(0).get_tex_symbol();
            auto s_j         = get_ro_edge_base(1).get_tex_symbol();
            auto Ttilde_sj   = get_ro_edge_base(2).get_tex_symbol();
            auto ds_j_dt     = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                NodeMonofluidTVASmoothSPositivityLimiter

                For gas particle $a$ and dust bin $j$:

                \begin{align}
                {ds_j_dt}_{j,a} &\leftarrow
                    \begin{cases}
                        0 & {s_j}_{j,a} < 0 \land {ds_j_dt}_{j,a} < 0 \\
                        {ds_j_dt}_{j,a} & \text{otherwise}
                    \end{cases} \\
                {ds_j_dt}_{j,a} &\mathrel{+}=
                    \begin{cases}
                        -{s_j}_{j,a} / (10\, {Ttilde_sj}_{j,a}) & {s_j}_{j,a} < 0 \\
                        0 & \text{otherwise}
                    \end{cases}
                \end{align}

                $a \in [0,{part_counts})$, $j \in [0,{ndust})$.
            )tex";

            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{s_j}", s_j);
            shambase::replace_all(tex, "{Ttilde_sj}", Ttilde_sj);
            shambase::replace_all(tex, "{ds_j_dt}", ds_j_dt);
            shambase::replace_all(tex, "{ndust}", sham::format("{}", ndust));

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
