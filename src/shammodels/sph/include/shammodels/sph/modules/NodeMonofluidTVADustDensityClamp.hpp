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
 * @file NodeMonofluidTVADustDensityClamp.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, hfactd)                                          \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, clamp_frac)                                      \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
                                                                                                   \
    /* inout */                                                                                    \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)

namespace shammodels::sph::modules {

    template<class Tvec>
    class NodeMonofluidTVADustDensityClamp : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        NodeMonofluidTVADustDensityClamp(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            Tscal pmass      = edges.gpart_mass.data;
            Tscal hfactd     = edges.hfactd.data;
            Tscal clamp_frac = edges.clamp_frac.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.hpart.get_spans()},
                sham::DDMultiRef{edges.s_j.get_spans()},
                edges.part_counts.indexes,
                [pmass, hfactd, clamp_frac, ndust = this->ndust](
                    u32 id_a, const Tscal *__restrict hpart, Tscal *__restrict s_j) {
                    u32 id_a_d = id_a * ndust;

                    Tscal h_a         = hpart[id_a];
                    Tscal rho_a       = shamrock::sph::rho_h(pmass, h_a, hfactd);
                    Tscal max_rho_d_j = clamp_frac * rho_a;

                    // pass 1 : clamp each species individually, accumulate the (post-clamp) sum
                    Tscal rho_d_sum = 0;
                    for (u32 j = 0; j < ndust; j++) {
                        Tscal sj     = s_j[id_a_d + j];
                        Tscal rho_dj = sj * sj / rho_a;
                        if (rho_dj > max_rho_d_j) {
                            rho_dj          = max_rho_d_j;
                            s_j[id_a_d + j] = sycl::sqrt(rho_dj * rho_a);
                        }
                        rho_d_sum += rho_dj;
                    }

                    // pass 2 : if the summed dust density still exceeds the threshold, scale
                    // every species down by the same factor so the sum lands at the threshold
                    if (rho_d_sum > max_rho_d_j && rho_d_sum > 0) {
                        Tscal scale = max_rho_d_j / rho_d_sum;
                        for (u32 j = 0; j < ndust; j++) {
                            Tscal sj        = s_j[id_a_d + j];
                            Tscal rho_dj    = sj * sj / rho_a;
                            Tscal rho_dj_sc = rho_dj * scale;
                            s_j[id_a_d + j] = sycl::sqrt(rho_dj_sc * rho_a);
                        }
                    }
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVADustDensityClamp";
        };

        inline virtual std::string _impl_get_tex() const {

            auto part_counts = get_ro_edge_base(0).get_tex_symbol();
            auto gpart_mass  = get_ro_edge_base(1).get_tex_symbol();
            auto hfactd      = get_ro_edge_base(2).get_tex_symbol();
            auto clamp_frac  = get_ro_edge_base(3).get_tex_symbol();
            auto hpart       = get_ro_edge_base(4).get_tex_symbol();
            auto s_j         = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                NodeMonofluidTVADustDensityClamp

                For gas particle $a$ and dust bin $j$, with
                $\rho_a = \rho({hpart}_a)$ the total density implied by the
                smoothing length and $f = {clamp_frac}$ the clamp fraction:

                \begin{align}
                {\rho_d}_{j,a} &= \min\left({s_j}_{j,a}^2 / \rho_a,\ f\,\rho_a\right) \\
                {\rho_d}_a &= \sum_j {\rho_d}_{j,a} \\
                {\rho_d}_{j,a} &\leftarrow
                    \begin{cases}
                        {\rho_d}_{j,a} \cdot f\,\rho_a / {\rho_d}_a
                            & {\rho_d}_a > f\,\rho_a \\
                        {\rho_d}_{j,a} & \text{otherwise}
                    \end{cases} \\
                {s_j}_{j,a} &\leftarrow \sqrt{{\rho_d}_{j,a}\, \rho_a}
                \end{align}

                $a \in [0,{part_counts})$, $j \in [0,{ndust})$.

                $m = {gpart_mass}$, $h_{{\rm fact}} = {hfactd}$.
            )tex";

            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{gpart_mass}", gpart_mass);
            shambase::replace_all(tex, "{hfactd}", hfactd);
            shambase::replace_all(tex, "{clamp_frac}", clamp_frac);
            shambase::replace_all(tex, "{hpart}", hpart);
            shambase::replace_all(tex, "{s_j}", s_j);
            shambase::replace_all(tex, "{ndust}", sham::format("{}", ndust));

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
