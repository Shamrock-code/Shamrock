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
 * @file AddForceShearingBoxNonInertial.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the inertial part of the acceleration for a shearing box force.
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {
    template<class Tvec>
    class AddForceShearingBoxNonInertial : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceShearingBoxNonInertial() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &omega_0;
            const shamrock::solvergraph::IDataEdge<Tscal> &q;

            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_velocities;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> omega_0,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> q,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_velocities,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges({omega_0, q, spans_positions, spans_velocities, sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(6),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(7),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal() {

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

        inline virtual std::string _impl_get_label() { return "AddForceShearingBoxNonInertial"; };

        virtual std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shammodels::common::modules
