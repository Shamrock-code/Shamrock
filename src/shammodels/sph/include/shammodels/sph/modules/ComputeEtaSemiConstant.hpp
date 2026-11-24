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
 * @file ComputeEtaSemiConstant.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Set the per-particle non-ideal MHD resistivities from the local state using the
 * "semi-constant" prescription of Wurster, Price & Bate (2016) (NICIL `icnstsemi`).
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, B_on_rho)                                        \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, eta_o)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, eta_h)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, eta_ad)

template<class Tvec, template<class> class SPHKernel>
class ComputeEtaSemiConstant : public shamrock::solvergraph::INode {

    using Tscal  = shambase::VecComponent<Tvec>;
    using Kernel = SPHKernel<Tscal>;

    Tscal part_mass;
    Tscal mu_0;
    Tscal C_O;
    Tscal C_H;
    Tscal C_AD;

    public:
    ComputeEtaSemiConstant(Tscal part_mass, Tscal mu_0, Tscal C_O, Tscal C_H, Tscal C_AD)
        : part_mass(part_mass), mu_0(mu_0), C_O(C_O), C_H(C_H), C_AD(C_AD) {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal pmass = part_mass;
        Tscal mu0   = mu_0;
        Tscal c_o   = C_O;
        Tscal c_h   = C_H;
        Tscal c_ad  = C_AD;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.B_on_rho.get_spans(), edges.hpart.get_spans()},
            sham::DDMultiRef{
                edges.eta_o.get_spans(), edges.eta_h.get_spans(), edges.eta_ad.get_spans()},
            edges.part_counts.indexes,
            [pmass, mu0, c_o, c_h, c_ad](
                u32 id_a,
                const Tvec *B_on_rho,
                const Tscal *hpart,
                Tscal *eta_o,
                Tscal *eta_h,
                Tscal *eta_ad) {
                Tscal rho_a = shamrock::sph::rho_h(pmass, hpart[id_a], Kernel::hfactd);
                Tvec B_a    = B_on_rho[id_a] * rho_a;
                Tscal B2_a  = sycl::dot(B_a, B_a);

                eta_o[id_a]  = c_o;
                eta_h[id_a]  = c_h * sycl::sqrt(B2_a / mu0);
                eta_ad[id_a] = c_ad * B2_a / (mu0 * rho_a);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeEtaSemiConstant"; };

    inline virtual std::string _impl_get_tex() const {
        std::string tex = R"tex(
            Semi-constant non-ideal MHD resistivities

            \begin{align}
            \rho_i &= m \left(\frac{h_{\rm fact}}{{hpart}_i}\right)^3 \\
            \mathbf{B}_i &= \rho_i {B_on_rho}_i \\
            {eta_o}_i &= C_O \\
            {eta_h}_i &= C_H \frac{\vert \mathbf{B}_i \vert}{\sqrt{\mu_0}} \\
            {eta_ad}_i &= C_{AD} \frac{\vert \mathbf{B}_i \vert^2}{\mu_0 \rho_i} \\
            i &\in [0,{part_counts})
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    };
};

#undef NODE_EDGES
