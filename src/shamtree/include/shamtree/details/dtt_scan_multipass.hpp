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
 * @file dtt_scan_multipass.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/details/reoder_scan_dtt_result.hpp"

namespace shamtree::details {

    template<class Tmorton, class Tvec, u32 dim>
    struct DTTScanMultipass {

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

            Tscal theta_sq = (s_a_sq + s_b_sq) / delta_ab_sq;

            return theta_sq < theta_crit * theta_crit;
        }

        inline static shamtree::DTTResult dtt(
            sham::DeviceScheduler_ptr dev_sched,
            const shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
            shambase::VecComponent<Tvec> theta_crit,
            bool ordered_result) {
            StackEntry stack_loc{};

            auto q = shambase::get_check_ref(dev_sched).get_queue();

            shamtree::DTTResult result{
                sham::DeviceBuffer<u32_2>(0, dev_sched), sham::DeviceBuffer<u32_2>(0, dev_sched)};

            sham::DeviceBuffer<u32_2> task_current(1, dev_sched);
            task_current.set_val_at_idx(0, {0, 0});

            u32 start_size = 1; // bvh.structure.get_total_cell_count();

            sham::DeviceBuffer<u32> has_pushed_task(start_size, dev_sched);
            sham::DeviceBuffer<u32_2> task_next(start_size, dev_sched);

            sham::DeviceBuffer<u32> has_pushed_m2m(start_size, dev_sched);
            sham::DeviceBuffer<u32_2> pushed_m2m(start_size, dev_sched);

            sham::DeviceBuffer<u32> has_pushed_p2p(start_size, dev_sched);
            sham::DeviceBuffer<u32_2> pushed_p2p(start_size, dev_sched);

            auto resize_max = [](auto &buf, u32 sz) {
                if (buf.get_size() < sz) {
                    buf.resize(sz);
                }
            };

            while (task_current.get_size() > 0) {
                u32 task_count = task_current.get_size();
                shamlog_debug_ln("dtt_scan_multipass", "task_current.get_size() :", task_count);

                // resizing BS
                u32 has_pushed_task_sz = task_count + 1;
                u32 task_next_sz       = 4 * task_count;
                u32 has_pushed_m2m_sz  = task_count + 1;
                u32 pushed_m2m_sz      = task_count;
                u32 has_pushed_p2p_sz  = task_count + 1;
                u32 pushed_p2p_sz      = task_count;

                resize_max(has_pushed_task, has_pushed_task_sz);
                resize_max(task_next, task_next_sz);
                resize_max(has_pushed_m2m, has_pushed_m2m_sz);
                resize_max(pushed_m2m, pushed_m2m_sz);
                resize_max(has_pushed_p2p, has_pushed_p2p_sz);
                resize_max(pushed_p2p, pushed_p2p_sz);

                has_pushed_task.fill(0, has_pushed_task_sz);
                has_pushed_m2m.fill(0, has_pushed_m2m_sz);
                has_pushed_p2p.fill(0, has_pushed_p2p_sz);

                using ObjectIterator  = shamtree::CLBVHObjectIterator<Tmorton, Tvec, dim>;
                ObjectIterator obj_it = bvh.get_object_iterator();

                using ObjItAcc = typename ObjectIterator::acc;

                // the embarrassingly parallel bit
                sham::kernel_call(
                    q,
                    sham::MultiRef{task_current, obj_it},
                    sham::MultiRef{
                        has_pushed_task,
                        task_next,
                        has_pushed_m2m,
                        pushed_m2m,
                        has_pushed_p2p,
                        pushed_p2p},
                    task_count,
                    [theta_crit](
                        u32 i,
                        const u32_2 *__restrict__ task_current,
                        ObjItAcc obj_it,
                        u32 *__restrict__ has_pushed_task,
                        u32_2 *__restrict__ task_next,
                        u32 *__restrict__ has_pushed_m2m,
                        u32_2 *__restrict__ pushed_m2m,
                        u32 *__restrict__ has_pushed_p2p,
                        u32_2 *__restrict__ pushed_p2p) {
                        u32_2 t = task_current[i];
                        u32 a   = t.x();
                        u32 b   = t.y();

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

                            if (child_a_1_leaf || child_a_2_leaf || child_b_1_leaf
                                || child_b_2_leaf) {
                                pushed_p2p[i]     = {a, b};
                                has_pushed_p2p[i] = 1;
                            } else {
                                task_next[i * 4 + 0] = {child_a_1, child_b_1};
                                task_next[i * 4 + 1] = {child_a_1, child_b_2};
                                task_next[i * 4 + 2] = {child_a_2, child_b_1};
                                task_next[i * 4 + 3] = {child_a_2, child_b_2};
                                has_pushed_task[i]   = 1;
                            }

                        } else {
                            pushed_m2m[i]     = {a, b};
                            has_pushed_m2m[i] = 1;
                        }
                    });

// set to false to use standard scans instead of in place ones
#if true
                shamalgs::primitives::scan_exclusive_sum_in_place(
                    has_pushed_task, has_pushed_task_sz);
                shamalgs::primitives::scan_exclusive_sum_in_place(
                    has_pushed_m2m, has_pushed_m2m_sz);
                shamalgs::primitives::scan_exclusive_sum_in_place(
                    has_pushed_p2p, has_pushed_p2p_sz);

#else
                has_pushed_task = shamalgs::numeric::scan_exclusive(
                    dev_sched, has_pushed_task, has_pushed_task_sz);
                has_pushed_m2m = shamalgs::numeric::scan_exclusive(
                    dev_sched, has_pushed_m2m, has_pushed_m2m_sz);
                has_pushed_p2p = shamalgs::numeric::scan_exclusive(
                    dev_sched, has_pushed_p2p, has_pushed_p2p_sz);
#endif
                sham::DeviceBuffer<u32> &scan_task = (has_pushed_task);
                sham::DeviceBuffer<u32> &scan_m2m  = (has_pushed_m2m);
                sham::DeviceBuffer<u32> &scan_p2p  = (has_pushed_p2p);

                // get the sizes of the result buffers before resizing
                u32 res_sz_node_node = result.node_interactions_m2m.get_size();
                u32 res_sz_leaf_leaf = result.node_interactions_p2p.get_size();

                // get the resulting count from the main kernel
                u32 count_task = scan_task.get_val_at_idx(has_pushed_task_sz - 1);
                u32 count_m2m  = scan_m2m.get_val_at_idx(has_pushed_m2m_sz - 1);
                u32 count_p2p  = scan_p2p.get_val_at_idx(has_pushed_p2p_sz - 1);

                // expand the result buffers
                result.node_interactions_m2m.expand(count_m2m);
                result.node_interactions_p2p.expand(count_p2p);

                // allocate space for the next pass
                task_current.resize(count_task * 4);

                // 4 wide stream compaction
                sham::kernel_call(
                    q,
                    sham::MultiRef{task_next, scan_task},
                    sham::MultiRef{task_current},
                    task_count,
                    [](u32 i,
                       const u32_2 *__restrict__ task_next,
                       const u32 *__restrict__ scan_task,
                       u32_2 *__restrict__ task_current) {
                        u32 scan_task_i   = scan_task[i];
                        u32 scan_task_ip1 = scan_task[i + 1];
                        if (scan_task_ip1 - scan_task_i == 1) {
                            u32 idx = scan_task_i * 4;

                            task_current[idx + 0] = task_next[i * 4 + 0];
                            task_current[idx + 1] = task_next[i * 4 + 1];
                            task_current[idx + 2] = task_next[i * 4 + 2];
                            task_current[idx + 3] = task_next[i * 4 + 3];
                        }
                    });

                // stream compaction
                sham::kernel_call(
                    q,
                    sham::MultiRef{pushed_m2m, scan_m2m},
                    sham::MultiRef{result.node_interactions_m2m},
                    task_count,
                    [res_sz_node_node](
                        u32 i,
                        const u32_2 *__restrict__ pushed_m2m,
                        const u32 *__restrict__ scan_m2m,
                        u32_2 *__restrict__ interacts_m2m) {
                        u32 scan_m2m_i   = scan_m2m[i];
                        u32 scan_m2m_ip1 = scan_m2m[i + 1];
                        if (scan_m2m_ip1 - scan_m2m_i == 1) {
                            interacts_m2m[res_sz_node_node + scan_m2m_i] = pushed_m2m[i];
                        }
                    });

                // stream compaction
                sham::kernel_call(
                    q,
                    sham::MultiRef{pushed_p2p, scan_p2p},
                    sham::MultiRef{result.node_interactions_p2p},
                    task_count,
                    [res_sz_leaf_leaf](
                        u32 i,
                        const u32_2 *__restrict__ pushed_p2p,
                        const u32 *__restrict__ scan_p2p,
                        u32_2 *__restrict__ interact_p2p) {
                        u32 scan_p2p_i   = scan_p2p[i];
                        u32 scan_p2p_ip1 = scan_p2p[i + 1];
                        if (scan_p2p_ip1 - scan_p2p_i == 1) {
                            interact_p2p[res_sz_leaf_leaf + scan_p2p_i] = pushed_p2p[i];
                        }
                    });
            }

            if (ordered_result) {
                DTTResult::OrderedResult ordering{
                    sham::DeviceBuffer<u32>(0, dev_sched), sham::DeviceBuffer<u32>(0, dev_sched)};

                shamtree::details::reorder_scan_dtt_result(
                    bvh.structure.get_total_cell_count(),
                    result.node_interactions_m2m,
                    ordering.offset_m2m);

                shamtree::details::reorder_scan_dtt_result(
                    bvh.structure.get_total_cell_count(),
                    result.node_interactions_p2p,
                    ordering.offset_p2p);

                result.ordered_result = std::move(ordering);
            }

            return result;
        }
    };
} // namespace shamtree::details
