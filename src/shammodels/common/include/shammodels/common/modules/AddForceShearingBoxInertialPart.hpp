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
 * @file AddForceShearingBoxInertialPart.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the inertial part of the acceleration for a shearing box force.
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
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {
    template<class Tvec>
    class AddForceShearingBoxInertialPart : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceShearingBoxInertialPart() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

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

        inline virtual std::string _impl_get_label() const {
            return "AddForceShearingBoxInertialPart";
        }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES
