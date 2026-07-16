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
 * @file AddForce1PN.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the 1PN force acceleration.
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_G)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_c)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, central_mass)                                    \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_pos)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_vel)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_velocities)                                \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {

    template<class Tvec>
    class AddForce1PN : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

    public:

        AddForce1PN() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal G       = edges.constant_G.data;
            Tscal c       = edges.constant_c.data;
            Tscal eps_gr = 1e-10; // small value to avoid division by zero
            Tscal cmass   = edges.central_mass.data;
            Tvec cpos     = edges.central_pos.data;
            Tvec cvel     = edges.central_vel.data;
            Tscal GM = cmass * G;


            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),

                sham::DDMultiRef{
                    edges.spans_positions.get_spans(),
                    edges.spans_velocities.get_spans()
                },

                sham::DDMultiRef{
                    edges.spans_accel_ext.get_spans()
                },

                edges.sizes.indexes,

                [cpos, cvel, GM, c, eps_gr](u32 gid,
                                            const Tvec *xyz,
                                            const Tvec *vxyz,
                                            Tvec *axyz_ext) {

                    Tvec r_a = xyz[gid] - cpos;
                    Tvec v_a = vxyz[gid] - cvel;

                    Tscal r = sycl::length(r_a);

                    Tvec r_hat = r_a / (r+eps_gr);

                    Tscal v2 = sham::dot(v_a, v_a);

                    Tscal vr = sham::dot(v_a, r_hat);


                    Tvec acc_1PN =
                        -GM / (r * r + eps_gr)
                        *
                        (
                            (
                                v2 / (c * c)
                                -
                                4 * GM / ((r + eps_gr) * c * c)
                            )
                            * r_hat

                            -

                            (4 * vr / (c * c))
                            * v_a
                        );


                    axyz_ext[gid] += acc_1PN;
                });
        }

        inline virtual std::string _impl_get_label() const { return "AddForce1PN"; };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::common::modules

#undef NODE_EDGES