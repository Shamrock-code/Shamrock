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
 * @file ForwardEuler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements a forward Euler integration step as a solver graph node.
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, time_derivative)                                    \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<T>, field)

namespace shammodels::common::modules {
    template<class T>
    class ForwardEulerPositive : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<T>;

        u32 nvar;

        public:
        ForwardEulerPositive(u32 nvar = 1) : nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.field.ensure_sizes(edges.sizes.indexes);

            Tscal dt = edges.dt.data;

            if (nvar == 1) {

                sham::distributed_data_kernel_call(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    sham::DDMultiRef{edges.time_derivative.get_spans()},
                    sham::DDMultiRef{edges.field.get_spans()},
                    edges.sizes.indexes,
                    [dt](u32 gid, const T *time_derivative, T *field) {
                        auto tmp = field[gid] + dt * time_derivative[gid];
                        field[gid] = std::max(tmp, 0.0);
                    });

            } else {

                auto var_count = edges.sizes.indexes.template map<u32>([&](u64 id, u32 count) {
                    return count * nvar;
                });

                sham::distributed_data_kernel_call(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    sham::DDMultiRef{edges.time_derivative.get_spans()},
                    sham::DDMultiRef{edges.field.get_spans()},
                    var_count,
                    [dt](u32 gid, const T *time_derivative, T *field) {
                        auto tmp = field[gid] + dt * time_derivative[gid];
                        field[gid] = std::max(tmp, 0.0);
                    });
            }
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEulerPositive"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES
