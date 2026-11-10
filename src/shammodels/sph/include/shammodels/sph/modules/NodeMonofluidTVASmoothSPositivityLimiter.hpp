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

        /// apply the xi switch (no further decrease of a negative s_j), disabled when the switch is
        /// already applied pairwise in the TVA derivative (pair gate), only phi is then applied
        bool apply_xi;

        public:
        NodeMonofluidTVASmoothSPositivityLimiter(u32 ndust, bool apply_xi = true)
            : ndust(ndust), apply_xi(apply_xi) {}

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
                [apply_xi = apply_xi](
                    u32 thread_id,
                    const Tscal *__restrict s_j,
                    const Tscal *__restrict Ttilde_sj,
                    Tscal *__restrict ds_j_dt) {
                    Tscal s_j_a       = s_j[thread_id];
                    Tscal Ttilde_sj_a = Ttilde_sj[thread_id];
                    Tscal ds_j_dt_a   = ds_j_dt[thread_id];

                    // if we dip in the negative range do not dip further
                    if (apply_xi) {
                        ds_j_dt_a *= (s_j_a < 0 && ds_j_dt_a < 0) ? 0 : 1;
                    }

                    // restore it slowly to 0
                    ds_j_dt_a += (s_j_a < 0) ? -s_j_a / (10 * Ttilde_sj_a) : 0;

                    ds_j_dt[thread_id] = ds_j_dt_a;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return apply_xi ? "NodeMonofluidTVASmoothSPositivityLimiter"
                            : "NodeMonofluidTVASmoothSPositivityLimiter (phi only)";
        };

        inline virtual std::string _impl_get_tex() const {
            std::string tex = R"tex(
                NodeMonofluidTVASmoothSPositivityLimiter

                For gas particle $a$ and dust bin $j$:

                \begin{align}
                {xi_switch}{ds_j_dt}_{j,a} &\mathrel{+}=
                    \begin{cases}
                        -{s_j}_{j,a} / (10\, {Ttilde_sj}_{j,a}) & {s_j}_{j,a} < 0 \\
                        0 & \text{otherwise}
                    \end{cases}
                \end{align}

                $a \in [0,{part_counts})$, $j \in [0,{ndust})$.
            )tex";

            const std::string xi_switch = apply_xi ? R"tex({ds_j_dt}_{j,a} &\leftarrow
                    \begin{cases}
                        0 & {s_j}_{j,a} < 0 \land {ds_j_dt}_{j,a} < 0 \\
                        {ds_j_dt}_{j,a} & \text{otherwise}
                    \end{cases} \\
                )tex"
                                                   : "";
            shambase::replace_all(tex, "{xi_switch}", xi_switch);

            replace_edges_tex_symbols(tex);

            shambase::replace_all(tex, "{ndust}", sham::format("{}", ndust));

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
