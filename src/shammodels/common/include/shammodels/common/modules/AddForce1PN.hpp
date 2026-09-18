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
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
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

            Tscal G     = edges.constant_G.data;
            Tscal c     = edges.constant_c.data;
            Tscal cmass = edges.central_mass.data;
            Tvec cpos   = edges.central_pos.data;
            Tvec cvel   = edges.central_vel.data;
            Tscal GM    = cmass * G;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),

                sham::DDMultiRef{
                    edges.spans_positions.get_spans(), edges.spans_velocities.get_spans()},

                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},

                edges.sizes.indexes,

                [cpos, cvel, GM, c](u32 gid, const Tvec *xyz, const Tvec *vxyz, Tvec *axyz_ext) {
                    Tvec r_a = xyz[gid] - cpos;
                    Tvec v_a = vxyz[gid] - cvel;

                    Tscal r      = sycl::length(r_a);
                    Tscal inv_r  = sham::inv_sat_zero(r);
                    Tscal inv_r2 = sham::inv_sat_zero(r * r);
                    Tvec r_hat   = r_a * inv_r;

                    Tscal v2 = sham::dot(v_a, v_a);

                    Tscal vr = sham::dot(v_a, r_hat);

                    Tvec acc_1PN = -GM * inv_r2
                                   * ((v2 / (c * c) - 4 * GM * inv_r / (c * c)) * r_hat

                                      -

                                      (4 * vr / (c * c)) * v_a);

                    axyz_ext[gid] += acc_1PN;
                });
        }

        inline virtual std::string _impl_get_label() const { return "AddForce1PN"; };

        inline virtual std::string _impl_get_tex() const {
            auto constant_G   = get_ro_edge_base(0).get_tex_symbol();
            auto constant_c   = get_ro_edge_base(1).get_tex_symbol();
            auto central_mass = get_ro_edge_base(2).get_tex_symbol();
            auto central_pos  = get_ro_edge_base(3).get_tex_symbol();
            auto central_vel  = get_ro_edge_base(4).get_tex_symbol();
            auto positions    = get_ro_edge_base(5).get_tex_symbol();
            auto velocities   = get_ro_edge_base(6).get_tex_symbol();
            auto axyz_ext     = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Add force (1PN)

                \begin{align}
                \mathbf{r}_i &= {positions}_i - {central_pos}_i\\
                \mathbf{v}_i &= {velocities}_i - {central_vel}_i\\
                r &= \sqrt{\sum_i r_i^2}\\
                \hat{\mathbf{r}}_i &= \mathbf{r}_i / r\\
                v^2 &= \sum_i v_i^2\\
                v_r &= \sum_i v_i \hat{\mathbf{r}}_i\\
                {axyz_ext}_i &\mathrel{+}= -\frac{{constant_G} {central_mass}}{r^2}
                \left[
                \left(\frac{v^2}{{constant_c}^2}
                - \frac{4 {constant_G} {central_mass}}{r {constant_c}^2}\right)\hat{\mathbf{r}}_i
                - \frac{4 v_r}{{constant_c}^2}\mathbf{v}_i
                \right]
                \end{align}
            )tex";

            shambase::replace_all(tex, "{constant_G}", constant_G);
            shambase::replace_all(tex, "{constant_c}", constant_c);
            shambase::replace_all(tex, "{central_mass}", central_mass);
            shambase::replace_all(tex, "{central_pos}", central_pos);
            shambase::replace_all(tex, "{central_vel}", central_vel);
            shambase::replace_all(tex, "{positions}", positions);
            shambase::replace_all(tex, "{velocities}", velocities);
            shambase::replace_all(tex, "{axyz_ext}", axyz_ext);

            return tex;
        };
    };

} // namespace shammodels::common::modules

#undef NODE_EDGES
