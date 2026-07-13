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
 * @file ComputeCFLDustDeltav.hpp
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
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, C_delta_v)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, cfl_density_threshold)                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, pmass)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, hfactd)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, delta_v)                                         \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tvec>
class ComputeCFLDustDeltav : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    u32 nbins;

    public:
    ComputeCFLDustDeltav(u32 nbins) : nbins(nbins) {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_delta_v             = edges.C_delta_v.value;
        Tscal cfl_density_threshold = edges.cfl_density_threshold.value;

        Tscal pmass  = edges.pmass.value;
        Tscal hfactd = edges.hfactd.value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.hpart.get_spans(), edges.delta_v.get_spans(), edges.s_j.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_delta_v, cfl_density_threshold, pmass, hfactd, nbins = this->nbins](
                u32 id_a,
                const Tscal *hpart,
                const Tvec *delta_v,
                const Tscal *s_j,
                Tscal *cfl_dt) {
                u32 id_a_d = id_a * nbins;

                Tscal h_a   = hpart[id_a];
                Tscal rho_a = shamrock::sph::rho_h(pmass, h_a, hfactd);

                auto rho_dust = [&](int j) {
                    auto tmp = s_j[id_a_d + j];
                    return tmp * tmp;
                };

                Tscal cfl_tmp = std::numeric_limits<Tscal>::infinity();

                for (int j = 0; j < nbins; j++) {
                    Tvec delta_v_j_a       = delta_v[id_a_d + j];
                    Tscal delta_v_j_a_norm = sycl::length(delta_v_j_a);

                    Tscal rho_d_j_a = rho_dust(j);
                    if (rho_d_j_a > cfl_density_threshold && delta_v_j_a_norm > 0) {
                        cfl_tmp = sycl::min(cfl_tmp, h_a / delta_v_j_a_norm);
                    }
                }

                cfl_tmp *= C_delta_v;

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], cfl_tmp);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLDustDeltav"; };

    inline virtual std::string _impl_get_tex() const { return "C_{\\Delta_v}"; };
};

#undef NODE_EDGES
