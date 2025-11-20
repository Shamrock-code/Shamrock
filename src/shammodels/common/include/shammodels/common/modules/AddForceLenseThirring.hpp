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
 * @file AddForceLenseThirring.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the Lense-Thirring force acceleration.
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
    class AddForceLenseThirring : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceLenseThirring() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_c;
            const shamrock::solvergraph::IDataEdge<Tscal> &central_mass;
            const shamrock::solvergraph::IDataEdge<Tvec> &central_pos;
            const shamrock::solvergraph::IDataEdge<Tscal> &a_spin;
            const shamrock::solvergraph::IDataEdge<Tvec> &dir_spin;

            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_velocities;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_c,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> central_mass,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tvec>> central_pos,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> a_spin,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tvec>> dir_spin,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_velocities,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges(
                {constant_G,
                 constant_c,
                 central_mass,
                 central_pos,
                 a_spin,
                 dir_spin,
                 spans_positions,
                 spans_velocities,
                 sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tvec>>(5),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(6),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(7),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(8),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal G       = edges.constant_G.data;
            Tscal c       = edges.constant_c.data;
            Tscal cmass   = edges.central_mass.data;
            Tvec cpos     = edges.central_pos.data;
            Tscal a_spin  = edges.a_spin.data;
            Tvec dir_spin = edges.dir_spin.data;

            Tscal GM = cmass * G;
            Tvec S   = a_spin * GM * GM * dir_spin / (c * c * c);

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    edges.spans_positions.get_spans(), edges.spans_velocities.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [cpos, S](u32 gid, const Tvec *xyz, const Tvec *vxyz, Tvec *axyz_ext) {
                    Tvec r_a       = xyz[gid];
                    Tvec v_a       = vxyz[gid];
                    Tscal abs_ra   = sycl::length(r_a);
                    Tscal abs_ra_2 = abs_ra * abs_ra;
                    Tscal abs_ra_3 = abs_ra_2 * abs_ra;
                    Tscal abs_ra_5 = abs_ra_2 * abs_ra_2 * abs_ra;

                    Tvec omega_a = (S * (2 / abs_ra_3)) - (6 * sham::dot(S, r_a) * r_a) / abs_ra_5;
                    Tvec acc_lt  = sycl::cross(v_a, omega_a);
                    axyz_ext[gid] += acc_lt;
                });
        }

        inline virtual std::string _impl_get_label() const { return "AddForceLenseThirring"; };

        virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules
