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
 * @file AddForceVerticalDiscPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the acceleration from a vertical disc potential.
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {
    template<class Tvec>
    class AddForceVerticalDiscPotential : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceVerticalDiscPotential() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shamrock::solvergraph::IDataEdge<Tscal> &central_mass;
            const shamrock::solvergraph::IDataEdge<Tscal> &R0;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> central_mass,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> R0,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges({constant_G, central_mass, R0, spans_positions, sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(4),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal cmass = edges.central_mass.data;
            Tscal G     = edges.constant_G.data;
            Tscal R0    = edges.R0.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_positions.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [mGM = -cmass * G, R02 = R0 * R0](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                    Tscal y_a = xyz[gid].y();
                    axyz_ext[gid].y() += mGM * y_a / sycl::sqrt(R02 + y_a * y_a);
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "AddForceVerticalDiscPotential";
        };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::common::modules
