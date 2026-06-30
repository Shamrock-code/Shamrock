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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
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
    class ForwardEuler : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<T>;

        u32 nvar;

        public:
        ForwardEuler(u32 nvar = 1) : nvar(nvar) {}

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
                        field[gid] = field[gid] + dt * time_derivative[gid];
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
                        field[gid] = field[gid] + dt * time_derivative[gid];
                    });
            }
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEuler"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

    // when optional nodes are made available, this can be merged with ForwardEuler
    template<class T>
    class ForwardEulerMasked : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<T>;

        public:
        ForwardEulerMasked() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &dt;
            const shamrock::solvergraph::IFieldSpan<T> &time_derivative;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<u32> &mask;
            shamrock::solvergraph::IFieldSpan<T> &field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> dt,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> time_derivative,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<u32>> mask,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> field) {
            __internal_set_ro_edges({dt, time_derivative, sizes, mask});
            __internal_set_rw_edges({field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(1),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<u32>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<T>>(0)};
        }

        void _impl_evaluate_internal() {
            __shamrock_stack_entry();
            auto edges = get_edges();
            edges.field.ensure_sizes(edges.sizes.indexes);
            edges.mask.check_sizes(edges.sizes.indexes);
            Tscal dt = edges.dt.data;
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.time_derivative.get_spans(), edges.mask.get_spans()},
                sham::DDMultiRef{edges.field.get_spans()},
                edges.sizes.indexes,
                [dt](u32 gid, const T *time_derivative, const u32 *mask, T *field) {
                    // Only update if mask is 0 (real particles), skip if mask is 1 (ghost
                    // particles)
                    if (mask[gid] == 0) {
                        field[gid] = field[gid] + dt * time_derivative[gid];
                    }
                });
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEulerMasked"; };
        virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES
