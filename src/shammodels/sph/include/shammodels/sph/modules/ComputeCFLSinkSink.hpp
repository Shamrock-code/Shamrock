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
 * @file ComputeCFLSinkSink.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Host-side CFL condition from the pairwise potential between sink particles.
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, G)                                               \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, C_force)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta_phi)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, positions)                           \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, masses)                             \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, acc_ext)                             \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, cfl_dt)

/**
 * @brief Host-side (N^2) sink-sink CFL condition.
 *
 * For each sink i with a non-zero external acceleration f_i, and each other sink j:
 * dt_ij = C_force * eta_phi * sqrt(|phi_ij| / |f_i|^2), with phi_ij = G * m_j / |r_ij|.
 * The result is the minimum over all pairs (infinity if no pair contributes).
 *
 * @tparam Tvec The sink position vector type
 */
template<class Tvec>
class ComputeCFLSinkSink : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    public:
    ComputeCFLSinkSink() = default;

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        Tscal G       = edges.G.data;
        Tscal C_force = edges.C_force.data;
        Tscal eta_phi = edges.eta_phi.data;

        const std::vector<Tvec> &pos     = edges.positions.data;
        const std::vector<Tscal> &mass   = edges.masses.data;
        const std::vector<Tvec> &acc_ext = edges.acc_ext.data;

        Tscal sink_sink_cfl = shambase::get_infty<Tscal>();

        for (u32 i = 0; i < pos.size(); i++) {
            Tscal sink_sink_cfl_i = shambase::get_infty<Tscal>();

            Tvec f_i = acc_ext[i];

            Tscal grad_phi_i_sq = sham::dot(f_i, f_i); // m^2.s^-4

            if (grad_phi_i_sq == 0) {
                continue;
            }

            for (u32 j = 0; j < pos.size(); j++) {
                if (i == j) {
                    continue;
                }

                Tvec rij       = pos[i] - pos[j];
                Tscal rij_scal = sycl::length(rij);

                Tscal phi_ij  = G * mass[j] / rij_scal;                  // J / kg = m^2.s^-2
                Tscal term_ij = sham::abs(phi_ij) / grad_phi_i_sq;       // s^2
                Tscal dt_ij   = C_force * eta_phi * sycl::sqrt(term_ij); // s

                sink_sink_cfl_i = sham::min(sink_sink_cfl_i, dt_ij);
            }

            sink_sink_cfl = sham::min(sink_sink_cfl, sink_sink_cfl_i);
        }

        edges.cfl_dt.data = sink_sink_cfl;
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLSinkSink"; };

    inline virtual std::string _impl_get_tex() const {
        std::string tex = R"tex(
            Sink-sink CFL

            \begin{align}
            \phi_{ij} &= \frac{{G} {masses}_j}{\vert {positions}_i - {positions}_j \vert} \\
            {cfl_dt} &= \min_{i, j \neq i, {acc_ext}_i \neq 0} {C_force} {eta_phi}
                \sqrt{\frac{\vert \phi_{ij} \vert}{{acc_ext}_i \cdot {acc_ext}_i}}
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    };
};

#undef NODE_EDGES
