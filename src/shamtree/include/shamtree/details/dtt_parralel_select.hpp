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
 * @file dtt_parralel_select.hpp
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

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTParallelSelect {

        using Tscal = shambase::VecComponent<Tvec>;

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

        inline static shamtree::DTTResult
        dtt(sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit) {
            StackEntry stack_loc{};

            auto q = shambase::get_check_ref(dev_sched).get_queue();

            using ObjectIterator  = shamtree::CLBVHObjectIterator<Tmorton, Tvec, dim>;
            ObjectIterator obj_it = bvh.get_object_iterator();

            using ObjItAcc = typename ObjectIterator::acc;

            u32 total_cell_count = bvh.structure.get_total_cell_count();

            sham::DeviceBuffer<u32> count_m2m(total_cell_count + 1, dev_sched);
            sham::DeviceBuffer<u32> count_p2p(total_cell_count + 1, dev_sched);
            count_m2m.set_val_at_idx(total_cell_count, 0);
            count_p2p.set_val_at_idx(total_cell_count, 0);

            // count the number of interactions for each cell

            sham::kernel_call(
                q,
                sham::MultiRef{obj_it},
                sham::MultiRef{count_m2m, count_p2p},
                total_cell_count,
                [theta_crit](
                    u32 i,
                    ObjItAcc obj_it,
                    u32 *__restrict__ count_m2m,
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

                    // push root-root interact on stack
                    stack.push({0, 0});

                    u32 count_m2m_i = 0;
                    u32 count_p2p_i = 0;

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
                            auto &ttrav = obj_it.tree_traverser.tree_traverser;

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

                        } else {
                            if (is_a_i_same) {
                                count_m2m_i++;
                            }
                        }
                    }

                    count_m2m[i] = count_m2m_i;
                    count_p2p[i] = count_p2p_i;
                });

            /////////////////////////////////////////////////////////////

            // scans the counts
            sham::DeviceBuffer<u32> scan_m2m
                = shamalgs::numeric::scan_exclusive(dev_sched, count_m2m, total_cell_count + 1);
            sham::DeviceBuffer<u32> scan_p2p
                = shamalgs::numeric::scan_exclusive(dev_sched, count_p2p, total_cell_count + 1);

            // alloc results buffers
            u32 total_count_m2m = scan_m2m.get_val_at_idx(total_cell_count);
            u32 total_count_p2p = scan_p2p.get_val_at_idx(total_cell_count);

            sham::DeviceBuffer<u32_2> idx_m2m(total_count_m2m, dev_sched);
            sham::DeviceBuffer<u32_2> idx_p2p(total_count_p2p, dev_sched);

            // relaunch the previous kernel but write the indexes this time

            sham::kernel_call(
                q,
                sham::MultiRef{obj_it, scan_m2m, scan_p2p},
                sham::MultiRef{idx_m2m, idx_p2p},
                total_cell_count,
                [theta_crit](
                    u32 i,
                    ObjItAcc obj_it,
                    const u32 *__restrict__ scan_m2m,
                    const u32 *__restrict__ scan_p2p,
                    u32_2 *__restrict__ idx_m2m,
                    u32_2 *__restrict__ idx_p2p) {
                    u32 offset_m2m = scan_m2m[i];
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

                    // push root-root interact on stack
                    stack.push({0, 0});

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
                            auto &ttrav = obj_it.tree_traverser.tree_traverser;

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

                        } else {
                            if (is_a_i_same) {
                                idx_m2m[offset_m2m] = {a, b};
                                offset_m2m++;
                            }
                        }
                    }
                });

            return DTTResult{std::move(idx_m2m), std::move(idx_p2p)};
        }
    };

} // namespace shamtree::details
