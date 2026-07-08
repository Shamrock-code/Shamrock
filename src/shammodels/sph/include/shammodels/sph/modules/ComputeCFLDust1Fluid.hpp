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
 * @file ComputeCFLDust1Fluid.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, C_1_fluid)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, pmass)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, hfactd)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, soundspeed)                                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, Ts_j)                                           \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tvec>
class ComputeCFLDust1Fluid : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    u32 nbins;

    public:
    ComputeCFLDust1Fluid(u32 nbins) : nbins(nbins) {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_1_fluid = edges.C_1_fluid.value;
        Tscal pmass     = edges.pmass.value;
        Tscal hfactd    = edges.hfactd.value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.hpart.get_spans(),
                edges.soundspeed.get_spans(),
                edges.s_j.get_spans(),
                edges.Ts_j.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_1_fluid, pmass, hfactd, nbins = this->nbins](
                u32 id_a,
                const Tscal *hpart,
                const Tscal *soundspeed,
                const Tscal *s_j,
                const Tscal *Ts_j,
                Tscal *cfl_dt) {
                u32 id_a_d = id_a * nbins;

                Tscal h_a   = hpart[id_a];
                Tscal rho_a = shamrock::sph::rho_h(pmass, h_a, hfactd);

                Tscal cs_a  = soundspeed[id_a];
                Tscal cs2_a = cs_a * cs_a;

                auto rho_dust = [&](int j) {
                    auto tmp = s_j[id_a_d + j];
                    return tmp * tmp;
                };

                auto epsilon_j = [&](int j) {
                    return rho_dust(j) / rho_a;
                };

                Tscal sum_eps = 0;
                for (int j = 0; j < nbins; j++) {
                    sum_eps += epsilon_j(j);
                }

                Tscal cs_tilde_2_a = cs2_a * (1 - sum_eps);

                Tscal cs4_over_h2 = cs2_a * cs2_a / (h_a * h_a);

                Tscal cfl_tmp = std::numeric_limits<Tscal>::infinity();

                for (int j = 0; j < nbins; j++) {
                    Tscal eps_j_a  = epsilon_j(j);
                    Tscal eps2_j_a = eps_j_a * eps_j_a;

                    Tscal Ts_j_a  = Ts_j[id_a_d + j];
                    Tscal Ts2_j_a = Ts_j_a * Ts_j_a;

                    Tscal dt_j = h_a / sycl::sqrt(cs_tilde_2_a + Ts2_j_a * eps2_j_a * cs4_over_h2);
                    cfl_tmp    = sycl::min(cfl_tmp, dt_j);
                }

                cfl_tmp *= C_1_fluid;

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], cfl_tmp);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLDust1Fluid"; };

    inline virtual std::string _impl_get_tex() const { return "C_{1,fluid}"; };
};

#undef NODE_EDGES
