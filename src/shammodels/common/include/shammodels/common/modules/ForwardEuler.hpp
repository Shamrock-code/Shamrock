// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ForwardEuler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the inertial part of the acceleration for a shearing box force.
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

namespace shammodels::common::modules {
    template<class T>
    class ForwardEuler : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<T>;

        public:
        ForwardEuler() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &dt;
            const shamrock::solvergraph::IFieldSpan<T> &time_derivative;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<T> &field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> dt,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> time_derivative,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> field) {
            __internal_set_ro_edges({dt, time_derivative, sizes});
            __internal_set_rw_edges({field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(1),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<T>>(0)};
        }

        void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.field.ensure_sizes(edges.sizes.indexes);

            Tscal dt = edges.dt.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.time_derivative.get_spans()},
                sham::DDMultiRef{edges.field.get_spans()},
                edges.sizes.indexes,
                [dt](u32 gid, const T *time_derivative, T *field) {
                    field[gid] = field[gid] + dt * time_derivative[gid];
                });
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEuler"; };

        virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules
