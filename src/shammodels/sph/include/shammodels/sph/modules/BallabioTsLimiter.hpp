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
 * @file BallabioTsLimiter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, cs)                                             \
                                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)

namespace shammodels::sph::modules {

    template<class Tvec>
    class BallabioTsLimiter : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        BallabioTsLimiter(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            edges.t_j.ensure_sizes(part_counts);

            auto &q = shamsys::instance::get_compute_scheduler().get_queue();

            part_counts.for_each([&](u64 id, u32 count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        edges.hpart.get_spans().get(id), edges.cs.get_spans().get(id)},
                    sham::MultiRef{edges.t_j.get_spans().get(id)},
                    count * ndust,
                    [ndust = ndust](
                        u32 thread_id,
                        const Tscal *__restrict hpart,
                        const Tscal *__restrict cs,
                        Tscal *__restrict t_j) {
                        u32 id_a = thread_id / ndust;

                        Tscal h_a  = hpart[id_a];
                        Tscal cs_a = cs[id_a];

                        t_j[thread_id] = sycl::min(t_j[thread_id], h_a / cs_a);
                    });
            });
        }

        inline virtual std::string _impl_get_label() const { return "BallabioTsLimiter"; };

        inline virtual std::string _impl_get_tex() const {

            auto part_counts = get_ro_edge_base(0).get_tex_symbol();
            auto hpart       = get_ro_edge_base(1).get_tex_symbol();
            auto cs          = get_ro_edge_base(2).get_tex_symbol();
            auto t_j         = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                BallabioTsLimiter

                \begin{align}
                {t_j}_{i,j} &= \min\left({t_j}_{i,j}, \frac{{hpart}_i}{{cs}_i}\right) \\
                i &\in [0,{part_counts}) \\
                j &\in [0,{ndust})
                \end{align}
            )tex";

            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{ndust}", shambase::format("{}", ndust));
            shambase::replace_all(tex, "{hpart}", hpart);
            shambase::replace_all(tex, "{cs}", cs);
            shambase::replace_all(tex, "{t_j}", t_j);

            return tex;
        };
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
