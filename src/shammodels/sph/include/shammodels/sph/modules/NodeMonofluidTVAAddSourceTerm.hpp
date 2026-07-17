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
 * @file NodeMonofluidTVAAddSourceTerm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, rhodust_eps)                                    \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt_hydro)                                       \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, S)                                              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, ds_j_dt)

namespace shammodels::sph::modules {

    template<class Tvec>
    class NodeMonofluidTVAAddSourceTerm : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 nbins;

        public:
        NodeMonofluidTVAAddSourceTerm(u32 nbins) : nbins(nbins) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.S.check_sizes(edges.part_counts.indexes);
            edges.s_j.check_sizes(edges.part_counts.indexes);
            edges.ds_j_dt.check_sizes(edges.part_counts.indexes);

            auto rhodust_eps = edges.rhodust_eps.value;

            shambase::DistributedData<u32> counts = edges.part_counts.indexes.template map<u32>(
                [nbins = this->nbins](u64 /**/, u32 count) -> u32 {
                    return count * nbins;
                });

            auto epsilon  = sycl::sqrt(rhodust_eps);
            auto dt_hydro = edges.dt_hydro.value;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.S.get_spans(), edges.s_j.get_spans()},
                sham::DDMultiRef{edges.ds_j_dt.get_spans()},
                counts,
                [rhodust_eps, epsilon, dt_hydro, nbins = this->nbins](
                    u32 id,
                    const Tscal *__restrict S,
                    const Tscal *__restrict s_j,
                    Tscal *__restrict ds_j_dt) {
                    Tscal sj = s_j[id];

                    Tscal ds_j_dt_val = S[id] / (2 * (sham::abs(sj) + epsilon));

                    if (sham::abs(dt_hydro * ds_j_dt_val) > 1e-2 * sham::abs(sj) || dt_hydro == 0) {
                        // here there is high likelihood of overshoot, so we go semi implicit
                        Tscal s_next = sycl::sqrt(sj * sj + S[id] * dt_hydro);
                        ds_j_dt_val  = (dt_hydro > 0) ? (s_next - sj) / dt_hydro : 0;
                    }

                    ds_j_dt[id] += ds_j_dt_val;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVAAddSourceTerm";
        };

        inline virtual std::string _impl_get_tex() const {

            auto S_edge           = get_ro_edge_base(2).get_tex_symbol();
            auto s_j_edge         = get_ro_edge_base(3).get_tex_symbol();
            auto ds_j_dt_edge     = get_rw_edge_base(0).get_tex_symbol();
            auto rhodust_eps_edge = get_ro_edge_base(1).get_tex_symbol();
            auto part_counts_edge = get_ro_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Monofluid TVA: dust-density source term $\rightarrow$ ${s_j}$ time derivative

                Per gas particle $a$ and mass bin $j$ (monofluid: $\rho_{{\rm d},j,a} = {s_j}_{j,a}^2$):

                \begin{align}
                \rho_{{\rm d},j,a} &= {s_j}_{j,a}^2 \\
                {S}_{j,a} &= \text{dust density source term } (\mathrm{d}\rho_{{\rm d},j,a}/\mathrm{d}t) \\
                \delta_{j,a} &= \begin{cases}
                    {S}_{j,a} / (2 |{s_j}_{j,a}|) & |{s_j}_{j,a}|^2 > \rho_{\rm eps} \\
                    {S}_{j,a} / \bigl(2 (|{s_j}_{j,a}| + \sqrt{\rho_{\rm eps}})\bigr) & \text{otherwise}
                \end{cases} \\
                {ds_j_dt}_{j,a} &\mathrel{+}= \delta_{j,a}
                \end{align}

                Unsaturated: $\mathrm{d}{s_j}_{j,a}^2/\mathrm{d}t = {S}_{j,a}
                \Rightarrow \mathrm{d}{s_j}_{j,a}/\mathrm{d}t = {S}_{j,a}/(2|{s_j}_{j,a}|)$.
                The floor ($\sqrt{\rho_{\rm eps}}$ in the denominator) lets bins with $|{s_j}_{j,a}|^2 \le \rho_{\rm eps}$ start accumulating.

                $a \in [0, {part_counts})$, $j \in [0, N_{\rm bins})$,
                $\rho_{\rm eps} = {rhodust_eps}$, $N_{\rm bins} = {nbins}$
            )tex";

            shambase::replace_all(tex, "{S}", S_edge);
            shambase::replace_all(tex, "{s_j}", s_j_edge);
            shambase::replace_all(tex, "{ds_j_dt}", ds_j_dt_edge);
            shambase::replace_all(tex, "{rhodust_eps}", rhodust_eps_edge);
            shambase::replace_all(tex, "{part_counts}", part_counts_edge);
            shambase::replace_all(tex, "{nbins}", shambase::format("{}", nbins));

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
