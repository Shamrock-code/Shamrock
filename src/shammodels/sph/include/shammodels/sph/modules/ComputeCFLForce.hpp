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
 * @file ComputeCFLForce.hpp
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
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, C_force)                                        \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, axyz)                                            \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, cfl_dt)

template<class Tvec>
class ComputeCFLForce : public shamrock::solvergraph::INode {

    using Tscal = shambase::VecComponent<Tvec>;

    public:
    ComputeCFLForce() {}

    EXPAND_NODE_EDGES(NODE_EDGES)

    inline void _impl_evaluate_internal() {
        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        Tscal C_force = edges.C_force.value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.hpart.get_spans(), edges.axyz.get_spans()},
            sham::DDMultiRef{edges.cfl_dt.get_spans()},
            edges.part_counts.indexes,
            [C_force](u32 id_a, const Tscal *hpart, const Tvec *axyz, Tscal *cfl_dt) {
                Tscal h_a     = hpart[id_a];
                Tscal abs_a_a = sycl::length(axyz[id_a]);

                Tscal dt_f = C_force * sycl::sqrt(h_a / abs_a_a);

                cfl_dt[id_a] = sycl::min(cfl_dt[id_a], dt_f);
            });
    }

    inline virtual std::string _impl_get_label() const { return "ComputeCFLForce"; };

    inline virtual std::string _impl_get_tex() const { return "C_{force}"; };
};

#undef NODE_EDGES
