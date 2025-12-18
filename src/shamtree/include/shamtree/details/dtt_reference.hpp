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

#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/details/multipole_acceptance_crit.hpp"
#include "shamtree/details/reorder_scan_dtt_result.hpp"

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTCpuReference {

        using Tscal = shambase::VecComponent<Tvec>;

        using ObjItHost    = shamtree::CLBVHObjectIteratorHost<Tmorton, Tvec, dim>;
        using ObjItHostAcc = typename ObjItHost::acc;

        inline static bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
            return shamtree::details::mac(a, b, theta_crit);
        }

        /// We make the assumption that the root is not a leaf
        template<bool allow_leaf_lowering>
        inline static void dtt_recursive_internal(
            u32 cell_a,
            u32 cell_b,
            const ObjItHostAcc &acc,
            Tscal theta_crit,
            std::vector<u32_2> &interact_m2l,
            std::vector<u32_2> &interact_p2p) {

            auto dtt_child_call = [&](u32 cell_a, u32 cell_b) {
                dtt_recursive_internal<allow_leaf_lowering>(
                    cell_a, cell_b, acc, theta_crit, interact_m2l, interact_p2p);
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

                if constexpr (allow_leaf_lowering) {

                    bool is_a_leaf = ttrav.is_id_leaf(cell_a);
                    bool is_b_leaf = ttrav.is_id_leaf(cell_b);

                    if (is_a_leaf && is_b_leaf) {
                        interact_p2p.push_back({cell_a, cell_b});
                        return;
                    }

                    u32 child_a_1 = (is_a_leaf) ? cell_a : ttrav.get_left_child(cell_a);
                    u32 child_a_2 = (is_a_leaf) ? cell_a : ttrav.get_right_child(cell_a);
                    u32 child_b_1 = (is_b_leaf) ? cell_b : ttrav.get_left_child(cell_b);
                    u32 child_b_2 = (is_b_leaf) ? cell_b : ttrav.get_right_child(cell_b);

                    bool run_a_1 = true;
                    bool run_a_2 = !is_a_leaf;
                    bool run_b_1 = true;
                    bool run_b_2 = !is_b_leaf;

                    if (run_a_1 && run_b_1)
                        dtt_child_call(child_a_1, child_b_1);
                    if (run_a_2 && run_b_1)
                        dtt_child_call(child_a_2, child_b_1);
                    if (run_a_1 && run_b_2)
                        dtt_child_call(child_a_1, child_b_2);
                    if (run_a_2 && run_b_2)
                        dtt_child_call(child_a_2, child_b_2);

                } else {

                    u32 child_a_1 = ttrav.get_left_child(cell_a);
                    u32 child_a_2 = ttrav.get_right_child(cell_a);
                    u32 child_b_1 = ttrav.get_left_child(cell_b);
                    u32 child_b_2 = ttrav.get_right_child(cell_b);

                    bool child_a_1_leaf = ttrav.is_id_leaf(child_a_1);
                    bool child_a_2_leaf = ttrav.is_id_leaf(child_a_2);
                    bool child_b_1_leaf = ttrav.is_id_leaf(child_b_1);
                    bool child_b_2_leaf = ttrav.is_id_leaf(child_b_2);

                    if (child_a_1_leaf || child_a_2_leaf || child_b_1_leaf || child_b_2_leaf) {
                        interact_p2p.push_back({cell_a, cell_b});
                        return;
                    }

                    dtt_child_call(child_a_1, child_b_1);
                    dtt_child_call(child_a_2, child_b_1);
                    dtt_child_call(child_a_1, child_b_2);
                    dtt_child_call(child_a_2, child_b_2);
                }

            } else {
                interact_m2l.push_back({cell_a, cell_b});
            }
        }

        inline static void dtt_recursive_ref(
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            Tscal theta_crit,
            std::vector<u32_2> &interact_m2l,
            std::vector<u32_2> &interact_p2p,
            bool allow_leaf_lowering) {

            __shamrock_stack_entry();

            auto obj_it_host = bvh.get_object_iterator_host();
            auto acc         = obj_it_host.get_read_access();

            auto &ttrav = acc.tree_traverser.tree_traverser;

            // Is the root a leaf ?
            if (ttrav.is_id_leaf(0)) {
                interact_p2p.push_back({0, 0});
                return;
            }

            /// We make the assumption that the root is not a leaf in this function
            if (allow_leaf_lowering) {
                dtt_recursive_internal<true>(0, 0, acc, theta_crit, interact_m2l, interact_p2p);
            } else {
                dtt_recursive_internal<false>(0, 0, acc, theta_crit, interact_m2l, interact_p2p);
            }
        }

        inline static shamtree::DTTResult dtt(
            sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit,
            bool ordered_result,
            bool allow_leaf_lowering) {
            StackEntry stack_loc{};

            std::vector<u32_2> interact_m2l{};
            std::vector<u32_2> interact_p2p{};

            dtt_recursive_ref(bvh, theta_crit, interact_m2l, interact_p2p, allow_leaf_lowering);

            sham::DeviceBuffer<u32_2> interact_m2l_buf(interact_m2l.size(), dev_sched);
            sham::DeviceBuffer<u32_2> interact_p2p_buf(interact_p2p.size(), dev_sched);

            interact_m2l_buf.copy_from_stdvec(interact_m2l);
            interact_p2p_buf.copy_from_stdvec(interact_p2p);

            // while we could have built the return object directly here, we instead build it
            // afterward to avoid an issue with clang-tidy complaining when initializing under the
            // hood multiple unique_ptr in a structured binding initialization
            // see : https://github.com/llvm/llvm-project/issues/153300
            DTTResult result{std::move(interact_m2l_buf), std::move(interact_p2p_buf)};

            if (ordered_result) {
                auto offset_m2l = sham::DeviceBuffer<u32>(0, dev_sched);
                auto offset_p2p = sham::DeviceBuffer<u32>(0, dev_sched);

                shamtree::details::reorder_scan_dtt_result(
                    bvh.structure.get_total_cell_count(), result.node_interactions_m2l, offset_m2l);

                shamtree::details::reorder_scan_dtt_result(
                    bvh.structure.get_total_cell_count(), result.node_interactions_p2p, offset_p2p);

                DTTResult::OrderedResult ordering{std::move(offset_m2l), std::move(offset_p2p)};

                result.ordered_result = std::move(ordering);
            }

            return result;
        }
    };
} // namespace shamtree::details
