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
 * @file dtt_parallel_select.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/FixedStack.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/details/multipole_acceptance_crit.hpp"

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTParallelSelect {

        using Tscal = shambase::VecComponent<Tvec>;

        inline static bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
            return shamtree::details::mac(a, b, theta_crit);
        }

        template<bool allow_leaf_lowering>
        inline static shamtree::DTTResult dtt_internal(
            sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit,
            bool ordered_result) {
            StackEntry stack_loc{};

            auto q = shambase::get_check_ref(dev_sched).get_queue();

            using ObjectIterator  = shamtree::CLBVHObjectIterator<Tmorton, Tvec, dim>;
            ObjectIterator obj_it = bvh.get_object_iterator();

            using ObjItAcc = typename ObjectIterator::acc;

            u32 total_cell_count = bvh.structure.get_total_cell_count();

            sham::DeviceBuffer<u32> count_m2l(total_cell_count + 1, dev_sched);
            sham::DeviceBuffer<u32> count_p2p(total_cell_count + 1, dev_sched);
            count_m2l.set_val_at_idx(total_cell_count, 0);
            count_p2p.set_val_at_idx(total_cell_count, 0);

            // count the number of interactions for each cell

            sham::kernel_call(
                q,
                sham::MultiRef{obj_it},
                sham::MultiRef{count_m2l, count_p2p},
                total_cell_count,
                [theta_crit](
                    u32 i,
                    ObjItAcc obj_it,
                    u32 *__restrict__ count_m2l,
                    u32 *__restrict__ count_p2p) {
                    shammath::AABB<Tvec> aabb_i
                        = {obj_it.tree_traverser.aabb_min[i], obj_it.tree_traverser.aabb_max[i]};

                    auto is_kdnode_within_node = [&](u32 node_id) -> bool {
                        shammath::AABB<Tvec> aabb_node
                            = {obj_it.tree_traverser.aabb_min[node_id],
                               obj_it.tree_traverser.aabb_max[node_id]};

                        return aabb_node.contains(aabb_i);
                    };

                    shambase::FixedStack<u32_2, ObjItAcc::tree_depth_max + 1> stack;

                    auto &ttrav = obj_it.tree_traverser.tree_traverser;

                    u32 count_m2l_i = 0;
                    u32 count_p2p_i = 0;

                    // Am I a leaf before we start going down the tree ?
                    if (i == 0 && ttrav.is_id_leaf(0)) {
                        count_p2p_i++;
                    } else {
                        // push root-root interact on stack
                        // We make the assumption that the root is not a leaf
                        stack.push({0, 0});
                    }

                    while (stack.is_not_empty()) {
                        u32_2 t = stack.pop_ret();
                        u32 a   = t.x();
                        u32 b   = t.y();

                        bool is_a_i_same = a == i;

                        shammath::AABB<Tvec> aabb_a = {
                            obj_it.tree_traverser.aabb_min[a], obj_it.tree_traverser.aabb_max[a]};
                        shammath::AABB<Tvec> aabb_b = {
                            obj_it.tree_traverser.aabb_min[b], obj_it.tree_traverser.aabb_max[b]};

                        bool crit = mac(aabb_a, aabb_b, theta_crit) == false;

                        if (crit) {

                            if constexpr (allow_leaf_lowering) {

                                bool is_a_leaf = ttrav.is_id_leaf(a);
                                bool is_b_leaf = ttrav.is_id_leaf(b);

                                if (is_a_leaf && is_b_leaf) {
                                    if (is_a_i_same) {
                                        count_p2p_i++;
                                    }
                                    continue;
                                }

                                u32 child_a_1 = (is_a_leaf) ? a : ttrav.get_left_child(a);
                                u32 child_a_2 = (is_a_leaf) ? a : ttrav.get_right_child(a);
                                u32 child_b_1 = (is_b_leaf) ? b : ttrav.get_left_child(b);
                                u32 child_b_2 = (is_b_leaf) ? b : ttrav.get_right_child(b);

                                bool run_a_1 = true;
                                bool run_a_2 = !is_a_leaf;
                                bool run_b_1 = true;
                                bool run_b_2 = !is_b_leaf;

                                // now since we can re-enqueue the same node we need to escape only
                                // if the child is enqueued so we replace the is_a_i_same condition
                                // from the case without lowering by is_a_i_same && (child_a_1 != a)
                                if (is_a_i_same && (child_a_1 != a)) {
                                    continue;
                                }

                                bool is_node_i_in_left_a  = is_kdnode_within_node(child_a_1);
                                bool is_node_i_in_right_a = is_kdnode_within_node(child_a_2);

                                run_a_1 = run_a_1 && is_node_i_in_left_a;
                                run_a_2 = run_a_2 && is_node_i_in_right_a;

                                if (run_a_1 && run_b_1)
                                    stack.push({child_a_1, child_b_1});
                                if (run_a_2 && run_b_1)
                                    stack.push({child_a_2, child_b_1});
                                if (run_a_1 && run_b_2)
                                    stack.push({child_a_1, child_b_2});
                                if (run_a_2 && run_b_2)
                                    stack.push({child_a_2, child_b_2});

                            } else {

                                u32 child_a_1 = ttrav.get_left_child(a);
                                u32 child_a_2 = ttrav.get_right_child(a);
                                u32 child_b_1 = ttrav.get_left_child(b);
                                u32 child_b_2 = ttrav.get_right_child(b);

                                bool child_a_1_leaf = ttrav.is_id_leaf(child_a_1);
                                bool child_a_2_leaf = ttrav.is_id_leaf(child_a_2);
                                bool child_b_1_leaf = ttrav.is_id_leaf(child_b_1);
                                bool child_b_2_leaf = ttrav.is_id_leaf(child_b_2);

                                if ((child_a_1_leaf || child_a_2_leaf || child_b_1_leaf
                                     || child_b_2_leaf)) {
                                    if (is_a_i_same) {
                                        count_p2p_i++; // found leaf-leaf interaction so skip child
                                                       // enqueue
                                    }
                                    continue;
                                }

                                bool is_node_i_in_left_a  = is_kdnode_within_node(child_a_1);
                                bool is_node_i_in_right_a = is_kdnode_within_node(child_a_2);

                                if (is_a_i_same) {
                                    continue;
                                }

                                if (is_node_i_in_left_a) {
                                    stack.push({child_a_1, child_b_1});
                                    stack.push({child_a_1, child_b_2});
                                }
                                if (is_node_i_in_right_a) {
                                    stack.push({child_a_2, child_b_1});
                                    stack.push({child_a_2, child_b_2});
                                }
                            }

                        } else {
                            if (is_a_i_same) {
                                count_m2l_i++;
                            }
                        }
                    }

                    count_m2l[i] = count_m2l_i;
                    count_p2p[i] = count_p2p_i;
                });

            /////////////////////////////////////////////////////////////

            // scans the counts
            sham::DeviceBuffer<u32> scan_m2l
                = shamalgs::numeric::scan_exclusive(dev_sched, count_m2l, total_cell_count + 1);
            sham::DeviceBuffer<u32> scan_p2p
                = shamalgs::numeric::scan_exclusive(dev_sched, count_p2p, total_cell_count + 1);

            // alloc results buffers
            u32 total_count_m2l = scan_m2l.get_val_at_idx(total_cell_count);
            u32 total_count_p2p = scan_p2p.get_val_at_idx(total_cell_count);

            sham::DeviceBuffer<u32_2> idx_m2l(total_count_m2l, dev_sched);
            sham::DeviceBuffer<u32_2> idx_p2p(total_count_p2p, dev_sched);

            // relaunch the previous kernel but write the indexes this time

            sham::kernel_call(
                q,
                sham::MultiRef{obj_it, scan_m2l, scan_p2p},
                sham::MultiRef{idx_m2l, idx_p2p},
                total_cell_count,
                [theta_crit](
                    u32 i,
                    ObjItAcc obj_it,
                    const u32 *__restrict__ scan_m2l,
                    const u32 *__restrict__ scan_p2p,
                    u32_2 *__restrict__ idx_m2l,
                    u32_2 *__restrict__ idx_p2p) {
                    u32 offset_m2l = scan_m2l[i];
                    u32 offset_p2p = scan_p2p[i];

                    shammath::AABB<Tvec> aabb_i
                        = {obj_it.tree_traverser.aabb_min[i], obj_it.tree_traverser.aabb_max[i]};

                    auto is_kdnode_within_node = [&](u32 node_id) -> bool {
                        shammath::AABB<Tvec> aabb_node
                            = {obj_it.tree_traverser.aabb_min[node_id],
                               obj_it.tree_traverser.aabb_max[node_id]};

                        return aabb_node.contains(aabb_i);
                    };

                    shambase::FixedStack<u32_2, ObjItAcc::tree_depth_max + 1> stack;

                    auto &ttrav = obj_it.tree_traverser.tree_traverser;

                    // Am I a leaf before we start going down the tree ?
                    if (i == 0 && ttrav.is_id_leaf(0)) {
                        idx_p2p[offset_p2p] = {0, 0};
                        offset_p2p++;
                    } else {
                        // push root-root interact on stack
                        // We make the assumption that the root is not a leaf
                        stack.push({0, 0});
                    }

                    while (stack.is_not_empty()) {
                        u32_2 t = stack.pop_ret();
                        u32 a   = t.x();
                        u32 b   = t.y();

                        bool is_a_i_same = a == i;

                        shammath::AABB<Tvec> aabb_a = {
                            obj_it.tree_traverser.aabb_min[a], obj_it.tree_traverser.aabb_max[a]};
                        shammath::AABB<Tvec> aabb_b = {
                            obj_it.tree_traverser.aabb_min[b], obj_it.tree_traverser.aabb_max[b]};

                        bool crit = mac(aabb_a, aabb_b, theta_crit) == false;

                        if (crit) {

                            if constexpr (allow_leaf_lowering) {

                                bool is_a_leaf = ttrav.is_id_leaf(a);
                                bool is_b_leaf = ttrav.is_id_leaf(b);

                                if (is_a_leaf && is_b_leaf) {
                                    if (is_a_i_same) {
                                        idx_p2p[offset_p2p] = {a, b};
                                        offset_p2p++;
                                    }
                                    continue;
                                }

                                u32 child_a_1 = (is_a_leaf) ? a : ttrav.get_left_child(a);
                                u32 child_a_2 = (is_a_leaf) ? a : ttrav.get_right_child(a);
                                u32 child_b_1 = (is_b_leaf) ? b : ttrav.get_left_child(b);
                                u32 child_b_2 = (is_b_leaf) ? b : ttrav.get_right_child(b);

                                bool run_a_1 = true;
                                bool run_a_2 = !is_a_leaf;
                                bool run_b_1 = true;
                                bool run_b_2 = !is_b_leaf;

                                // now since we can re-enqueue the same node we need to escape only
                                // if the child is enqueued so we replace the is_a_i_same condition
                                // from the case without lowering by is_a_i_same && (child_a_1 != a)
                                if (is_a_i_same && (child_a_1 != a)) {
                                    continue;
                                }

                                bool is_node_i_in_left_a  = is_kdnode_within_node(child_a_1);
                                bool is_node_i_in_right_a = is_kdnode_within_node(child_a_2);

                                run_a_1 = run_a_1 && is_node_i_in_left_a;
                                run_a_2 = run_a_2 && is_node_i_in_right_a;

                                if (run_a_1 && run_b_1)
                                    stack.push({child_a_1, child_b_1});
                                if (run_a_2 && run_b_1)
                                    stack.push({child_a_2, child_b_1});
                                if (run_a_1 && run_b_2)
                                    stack.push({child_a_1, child_b_2});
                                if (run_a_2 && run_b_2)
                                    stack.push({child_a_2, child_b_2});

                            } else {
                                u32 child_a_1 = ttrav.get_left_child(a);
                                u32 child_a_2 = ttrav.get_right_child(a);
                                u32 child_b_1 = ttrav.get_left_child(b);
                                u32 child_b_2 = ttrav.get_right_child(b);

                                bool child_a_1_leaf = ttrav.is_id_leaf(child_a_1);
                                bool child_a_2_leaf = ttrav.is_id_leaf(child_a_2);
                                bool child_b_1_leaf = ttrav.is_id_leaf(child_b_1);
                                bool child_b_2_leaf = ttrav.is_id_leaf(child_b_2);

                                if ((child_a_1_leaf || child_a_2_leaf || child_b_1_leaf
                                     || child_b_2_leaf)) {
                                    if (is_a_i_same) {
                                        idx_p2p[offset_p2p] = {a, b};
                                        offset_p2p++;
                                    }
                                    continue;
                                }

                                bool is_node_i_in_left_a  = is_kdnode_within_node(child_a_1);
                                bool is_node_i_in_right_a = is_kdnode_within_node(child_a_2);

                                if (is_a_i_same) {
                                    continue;
                                }

                                if (is_node_i_in_left_a) {
                                    stack.push({child_a_1, child_b_1});
                                    stack.push({child_a_1, child_b_2});
                                }
                                if (is_node_i_in_right_a) {
                                    stack.push({child_a_2, child_b_1});
                                    stack.push({child_a_2, child_b_2});
                                }
                            }

                        } else {
                            if (is_a_i_same) {
                                idx_m2l[offset_m2l] = {a, b};
                                offset_m2l++;
                            }
                        }
                    }
                });

            DTTResult ret{std::move(idx_m2l), std::move(idx_p2p)};

            if (ordered_result) {
                DTTResult::OrderedResult ordering{std::move(scan_m2l), std::move(scan_p2p)};
                ret.ordered_result = std::move(ordering);
            }

            return ret;
        }

        inline static shamtree::DTTResult dtt(
            sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit,
            bool ordered_result,
            bool allow_leaf_lowering) {
            if (allow_leaf_lowering) {
                return dtt_internal<true>(dev_sched, bvh, theta_crit, ordered_result);
            } else {
                return dtt_internal<false>(dev_sched, bvh, theta_crit, ordered_result);
            }
        }
    };

} // namespace shamtree::details
