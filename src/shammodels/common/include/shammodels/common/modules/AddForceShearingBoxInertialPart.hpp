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
 * @file AddForceShearingBoxInertialPart.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
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
    template<class Tvec>
    class AddForceShearingBoxInertialPart : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceShearingBoxInertialPart() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &eta;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> eta,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges({eta, spans_positions, sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(1),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal eta = edges.eta.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_positions.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [two_eta = 2 * eta](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                    Tvec r_a = xyz[gid];
                    axyz_ext[gid] += Tvec{r_a.x() * two_eta, 0, 0};
                });
        }

        inline virtual std::string _impl_get_label() { return "AddForceShearingBoxInertialPart"; };

        virtual std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shammodels::common::modules
