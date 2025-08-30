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
 * @file dtt_reference.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTCpuReference {

        using Tscal = shambase::VecComponent<Tvec>;

        using ObjItHost    = shamtree::CLBVHObjectIteratorHost<Tmorton, Tvec, 3>;
        using ObjItHostAcc = typename ObjItHost::acc;

        inline static bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
            Tvec s_a      = (a.upper - a.lower);
            Tvec s_b      = (b.upper - b.lower);
            Tvec r_a      = (a.upper + a.lower) / 2;
            Tvec r_b      = (b.upper + b.lower) / 2;
            Tvec delta_ab = r_a - r_b;

            Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

            if (delta_ab_sq == 0) {
                return false;
            }

            Tscal s_a_sq = sham::dot(s_a, s_a);
            Tscal s_b_sq = sham::dot(s_b, s_b);

            Tscal r_a_sq = sham::dot(r_a, r_a);
            Tscal r_b_sq = sham::dot(r_b, r_b);

            Tscal theta_sq = (s_a_sq + s_b_sq) / delta_ab_sq;

            return theta_sq < theta_crit * theta_crit;
        }

        inline static void dtt_recursive_internal(
            u32 cell_a,
            u32 cell_b,
            const ObjItHostAcc &acc,
            Tscal theta_crit,
            std::vector<u32_2> &interact_p2p,
            std::vector<u32_2> &interact_m2m) {

            auto dtt_child_call = [&](u32 cell_a, u32 cell_b) {
                dtt_recursive_internal(cell_a, cell_b, acc, theta_crit, interact_p2p, interact_m2m);
            };

            auto &ttrav = acc.tree_traverser.tree_traverser;

            Tvec aabb_min_a = acc.tree_traverser.aabb_min[cell_a];
            Tvec aabb_max_a = acc.tree_traverser.aabb_max[cell_a];

            Tvec aabb_min_b = acc.tree_traverser.aabb_min[cell_b];
            Tvec aabb_max_b = acc.tree_traverser.aabb_max[cell_b];

            shammath::AABB<Tvec> aabb_a = {aabb_min_a, aabb_max_a};
            shammath::AABB<Tvec> aabb_b = {aabb_min_b, aabb_max_b};

            bool crit = mac(aabb_a, aabb_b, theta_crit) == false;

            if (crit) {

                u32 child_a_1 = ttrav.get_left_child(cell_a);
                u32 child_a_2 = ttrav.get_right_child(cell_a);
                u32 child_b_1 = ttrav.get_left_child(cell_b);
                u32 child_b_2 = ttrav.get_right_child(cell_b);

                bool child_a_1_leaf = ttrav.is_id_leaf(child_a_1);
                bool child_a_2_leaf = ttrav.is_id_leaf(child_a_2);
                bool child_b_1_leaf = ttrav.is_id_leaf(child_b_1);
                bool child_b_2_leaf = ttrav.is_id_leaf(child_b_2);

                if (child_a_1_leaf || child_a_2_leaf || child_b_1_leaf || child_b_2_leaf) {
                    interact_m2m.push_back({cell_a, cell_b});
                    return;
                }

                dtt_child_call(child_a_1, child_b_1);
                dtt_child_call(child_a_2, child_b_1);
                dtt_child_call(child_a_1, child_b_2);
                dtt_child_call(child_a_2, child_b_2);

            } else {
                interact_p2p.push_back({cell_a, cell_b});
            }
        }

        inline static void dtt_recursive_ref(
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, 3> &bvh,
            Tscal theta_crit,
            std::vector<u32_2> &interact_p2p,
            std::vector<u32_2> &interact_m2m) {

            auto obj_it_host = bvh.get_object_iterator_host();
            auto acc         = obj_it_host.get_read_access();

            dtt_recursive_internal(0, 0, acc, theta_crit, interact_p2p, interact_m2m);
        }

        inline static shamtree::DTTResult
        dtt(sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit) {
            StackEntry stack_loc{};

            std::vector<u32_2> interact_m2m{};
            std::vector<u32_2> interact_p2p{};

            dtt_recursive_ref(bvh, theta_crit, interact_m2m, interact_p2p);

            shamtree::DTTResult result{
                sham::DeviceBuffer<u32_2>(interact_m2m.size(), dev_sched),
                sham::DeviceBuffer<u32_2>(interact_p2p.size(), dev_sched)};

            result.node_interactions_m2m.copy_from_stdvec(interact_m2m);
            result.node_interactions_p2p.copy_from_stdvec(interact_p2p);

            return result;
        }
    };
} // namespace shamtree::details
