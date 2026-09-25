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
 * @file ComputeCFLNIMHDVaryingEta.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief CFL limiter for non-ideal MHD when etaO/etaH/etaAD are set per-particle
 *        (SolverConfig::has_field_eta()) instead of being global constants.
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, C_nimhd)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, eta_o)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, eta_ad)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, eta_h)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tvec>
class ComputeCFLNIMHDVaryingEta : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    public:
    ComputeCFLNIMHDVaryingEta() {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_nimhd = edges.C_nimhd.data;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.hpart.get_spans(),
                edges.eta_o.get_spans(),
                edges.eta_ad.get_spans(),
                edges.eta_h.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_nimhd](
                u32 id_a,
                const Tscal *hpart,
                const Tscal *eta_o,
                const Tscal *eta_ad,
                const Tscal *eta_h,
                Tscal *cfl_dt) {
                Tscal h_a     = hpart[id_a];
                Tscal max_eta = sycl::max(
                    sycl::max(sycl::fabs(eta_o[id_a]), sycl::fabs(eta_ad[id_a])),
                    sycl::fabs(eta_h[id_a]));

                Tscal dt_nimhd = C_nimhd * h_a * h_a / max_eta;

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], dt_nimhd);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLNIMHDVaryingEta"; };

    inline virtual std::string _impl_get_tex() const { return "C_{NIMHD}"; };
};

#undef NODE_EDGES
