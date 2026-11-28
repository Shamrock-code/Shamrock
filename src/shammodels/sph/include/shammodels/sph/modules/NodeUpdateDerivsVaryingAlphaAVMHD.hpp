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
 * @file NodeUpdateDerivsVaryingAlphaAVMHD.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Hydro update derivs (pressure, artificial viscosity & conductivity) with a varying alpha
 * AV, using the MHD signal velocity (fast magnetosonic speed instead of the sound speed).
 *
 */

#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, alpha_u)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, beta_AV)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, mu_0)                                            \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts_with_ghost)                              \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, xyz)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, vxyz)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, uint)                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, omega)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, pressure)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, cs)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, alpha_AV)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, B_on_rho)                                        \
                                                                                                   \
    /* neigh */                                                                                    \
    X_RO(shammodels::sph::solvergraph::NeighCache, neigh_cache)                                    \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, axyz)                                            \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, duint)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class NodeUpdateDerivsVaryingAlphaAVMHD : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;

        public:
        NodeUpdateDerivsVaryingAlphaAVMHD() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "UpdateDerivsVaryingAlphaAVMHD";
        };

        inline virtual std::string _impl_get_tex() const {
            std::string tex = R"tex(
            Hydro update derivs with varying $\alpha$ artificial viscosity, using the MHD signal
            velocity (Price et al. 2018, eqs.~34, 35, 40, 42 with $c_s \to v_f$). The pure MHD terms
            are added separately.

            \begin{align}
            \rho_a &= {gpart_mass} \left( \frac{h_{\rm fact}}{ {hpart}_a } \right)^3,
                \quad \mathbf{B}_a = \rho_a {B_on_rho}_a \\
            v_{f,a} &= \sqrt{ {cs}_a^2 + \frac{\vert \mathbf{B}_a \vert^2}{ {mu_0} \rho_a } } \\
            \mathbf{v}_{ab} &= {vxyz}_a - {vxyz}_b, \quad
                \hat{\mathbf{r}}_{ab} = \frac{ {xyz}_a - {xyz}_b }{ \vert {xyz}_a - {xyz}_b \vert },
                \quad F_{ab}(h) = \frac{\partial W}{\partial r}(\vert {xyz}_a - {xyz}_b \vert, h) \\
            v_{{\rm sig},a} &= {alpha_AV}_a v_{f,a}
                + {beta_AV} \vert \mathbf{v}_{ab} \cdot \hat{\mathbf{r}}_{ab} \vert \\
            q^a_{ab} &= \max\left( -\frac{1}{2} \rho_a v_{{\rm sig},a}
                \, \mathbf{v}_{ab} \cdot \hat{\mathbf{r}}_{ab}, 0 \right) \\
            v^u_{\rm sig} &= \sqrt{ \frac{ 2 \vert {pressure}_a - {pressure}_b \vert }{ \rho_a + \rho_b } } \\
            {axyz}_a &= - {gpart_mass} \sum_b \left[
                \frac{ {pressure}_a + q^a_{ab} }{ {omega}_a \rho_a^2 } F_{ab}(h_a)
                + \frac{ {pressure}_b + q^b_{ab} }{ {omega}_b \rho_b^2 } F_{ab}(h_b)
                \right] \hat{\mathbf{r}}_{ab} \\
            {duint}_a &= {gpart_mass} \sum_b \left[
                \frac{ {pressure}_a + q^a_{ab} }{ {omega}_a \rho_a^2 }
                \, \mathbf{v}_{ab} \cdot \hat{\mathbf{r}}_{ab} \, F_{ab}(h_a)
                + \frac{1}{2} {alpha_u} \, v^u_{\rm sig} ({uint}_a - {uint}_b)
                \left( \frac{F_{ab}(h_a)}{ {omega}_a \rho_a } + \frac{F_{ab}(h_b)}{ {omega}_b \rho_b } \right)
                \right] \\
            a &\in [0, {part_counts}), \quad h_{\rm fact} = {hfact}
            \end{align}
            )tex";

            replace_edges_tex_symbols(tex);

            shambase::replace_all(tex, "{hfact}", sham::format("{}", SPHKernel<Tscal>::hfactd));

            return tex;
        };
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
