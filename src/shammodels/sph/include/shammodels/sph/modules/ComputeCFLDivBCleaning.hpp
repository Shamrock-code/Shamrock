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
 * @file ComputeCFLDivBCleaning.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, C_cour)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, vclean)                                         \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tscal>
class ComputeCFLDivBCleaning : public shamrock::solvergraph::INode {

    public:
    ComputeCFLDivBCleaning() {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_cour = edges.C_cour.value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.hpart.get_spans(), edges.vclean.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_cour](u32 id_a, const Tscal *hpart, const Tscal *vclean, Tscal *cfl_dt) {
                Tscal h_a      = hpart[id_a];
                Tscal vclean_a = vclean[id_a];

                Tscal dt_divB_cleaning = C_cour * h_a / vclean_a;

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], dt_divB_cleaning);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLDivBCleaning"; };

    inline virtual std::string _impl_get_tex() const { return "C_{divB_cleaning}"; };
};

#undef NODE_EDGES
