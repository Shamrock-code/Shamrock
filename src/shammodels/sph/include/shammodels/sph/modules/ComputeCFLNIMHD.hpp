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
 * @file ComputeCFLNIMHD.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
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
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta_O)                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta_AD)                                          \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta_H)                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tvec>
class ComputeCFLNIMHD : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    public:
    ComputeCFLNIMHD() {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_nimhd = edges.C_nimhd.data;
        Tscal eta_O   = edges.eta_O.data;
        Tscal eta_AD  = edges.eta_AD.data;
        Tscal eta_H   = edges.eta_H.data;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.hpart.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_nimhd, eta_O, eta_AD, eta_H](u32 id_a, const Tscal *hpart, Tscal *cfl_dt) {
                Tscal h_a     = hpart[id_a];
                Tscal max_eta = sycl::max(sycl::max(eta_O, eta_AD), eta_H);

                Tscal dt_nimhd = C_nimhd * h_a * h_a / max_eta;

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], dt_nimhd);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLNIMHD"; };

    inline virtual std::string _impl_get_tex() const { return "C_{NIMHD}"; };
};

#undef NODE_EDGES
