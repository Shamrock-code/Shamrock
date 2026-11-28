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
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, rhodust_eps)                                     \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, tau_frac)                                        \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt_hydro)                                        \
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

            auto rhodust_eps = edges.rhodust_eps.data;

            shambase::DistributedData<u32> counts = edges.part_counts.indexes.template map<u32>(
                [nbins = this->nbins](u64 /**/, u32 count) -> u32 {
                    return count * nbins;
                });

            auto dt_hydro = edges.dt_hydro.data;
            auto drho_min = edges.tau_frac.data * rhodust_eps;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.S.get_spans(), edges.s_j.get_spans()},
                sham::DDMultiRef{edges.ds_j_dt.get_spans()},
                counts,
                [drho_min, dt_hydro, nbins = this->nbins](
                    u32 id,
                    const Tscal *__restrict S,
                    const Tscal *__restrict s_j,
                    Tscal *__restrict ds_j_dt) {
                    Tscal src = S[id];

                    // no source, no contribution (and dt_eff below would be infinite)
                    if (src == 0) {
                        return;
                    }

                    Tscal sj = s_j[id];

                    // implicit step with a floor on dt: dividing by a tiny dt_hydro (e.g. right
                    // before a normal step) would give a very large derivative, that explodes at
                    // the next step when the leapfrog applies it over a larger dt.

                    // dt_eps = time to fill the fraction tau_frac of rho_eps at the current rate
                    Tscal dt_eps = drho_min / sham::abs(src);
                    Tscal dt_eff = sham::max(dt_hydro, dt_eps);
                    Tscal s_next = sycl::sqrt(sham::max(sj * sj + src * dt_eff, Tscal(0)));

                    ds_j_dt[id] += (s_next - sj) / dt_eff;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVAAddSourceTerm";
        };

        inline virtual std::string _impl_get_tex() const {
            std::string tex = R"tex(
                Monofluid TVA: dust-density source term $\rightarrow$ ${s_j}$ time derivative

                Per gas particle $a$ and mass bin $j$ (monofluid: $\rho_{{\rm d},j,a} = {s_j}_{j,a}^2$):

                \begin{align}
                \rho_{{\rm d},j,a} &= {s_j}_{j,a}^2 \\
                {S}_{j,a} &= \text{dust density source term } (\mathrm{d}\rho_{{\rm d},j,a}/\mathrm{d}t) \\
                \Delta t_{{\rm eff},j,a} &= \max\bigl({dt_hydro}, {tau_frac}\, \rho_{\rm eps} / |{S}_{j,a}|\bigr) \\
                \delta_{j,a} &= \Bigl(\sqrt{\max({s_j}_{j,a}^2 + {S}_{j,a} \Delta t_{{\rm eff},j,a}, 0)}
                    - {s_j}_{j,a}\Bigr) / \Delta t_{{\rm eff},j,a} \\
                {ds_j_dt}_{j,a} &\mathrel{+}= \delta_{j,a}
                \end{align}

                Implicit in $\rho_{{\rm d},j,a}$: $\delta_{j,a}$ is the mean rate of ${s_j}_{j,a}$ over
                $\Delta t_{\rm eff}$ when $\rho_{{\rm d},j,a}$ moves by ${S}_{j,a} \Delta t_{\rm eff}$
                (clamped at $0$), which tends to ${S}_{j,a}/(2|{s_j}_{j,a}|)$ for small steps.
                $\Delta t_{\rm eff}$ moves at least ${tau_frac}\, \rho_{\rm eps}$, so that a null or tiny
                ${dt_hydro}$ cannot give a large rate for an empty bin.
                ${S}_{j,a} = 0$ gives $\delta_{j,a} = 0$.

                $a \in [0, {part_counts})$, $j \in [0, N_{\rm bins})$,
                $\rho_{\rm eps} = {rhodust_eps}$, $N_{\rm bins} = {nbins}$
            )tex";

            replace_edges_tex_symbols(tex);

            shambase::replace_all(tex, "{nbins}", sham::format("{}", nbins));

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
