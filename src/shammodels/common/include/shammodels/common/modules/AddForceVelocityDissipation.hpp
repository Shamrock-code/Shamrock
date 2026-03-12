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
 * @file AddForceVelocityDissipation.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the acceleration from a velocity dissipation force.
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

/// declare the list of edges for this node
#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_velocity)                                  \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {
    template<class Tvec>
    class AddForceVelocityDissipation : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceVelocityDissipation() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_velocity.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal eta = edges.eta.data;

            // call the GPU kernel for each patches
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_velocity.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [eta](u32 gid, const Tvec *vxyz, Tvec *axyz_ext) {
                    axyz_ext[gid] -= eta * vxyz[gid];
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "AddForceVelocityDissipation";
        };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::common::modules

// remove the macro before exiting
#undef NODE_EDGES
