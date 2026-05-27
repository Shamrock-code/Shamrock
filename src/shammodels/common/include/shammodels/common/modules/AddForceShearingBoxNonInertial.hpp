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
 * @file AddForceShearingBoxNonInertial.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the non-inertial part of the acceleration for a shearing box force.
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, omega_0)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, q)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_velocities)                                \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {

    template<class Tvec>
    class AddForceShearingBoxNonInertial : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceShearingBoxNonInertial() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal Omega_0 = edges.omega_0.data;
            Tscal q       = edges.q.data;

            Tscal Omega_0_sq = Omega_0 * Omega_0;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    edges.spans_positions.get_spans(), edges.spans_velocities.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [Omega_0, Omega_0_sq, q](
                    u32 gid, const Tvec *xyz, const Tvec *vxyz, Tvec *axyz_ext) {
                    Tvec r_a = xyz[gid];
                    Tvec v_a = vxyz[gid];
                    axyz_ext[gid] += Tvec{
                        2 * Omega_0 * (q * Omega_0 * r_a.x() + v_a.y()),
                        -2 * Omega_0 * v_a.x(),
                        -Omega_0_sq * r_a.z()};
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "AddForceShearingBoxNonInertial";
        }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES
