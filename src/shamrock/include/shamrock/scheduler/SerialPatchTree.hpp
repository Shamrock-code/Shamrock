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
 * @file SerialPatchTree.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

//%Impl status : Should rewrite

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/BufferMirror.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include <array>
#include <optional>
#include <tuple>
#include <vector>

template<class fp_prec_vec>
class SerialPatchTree {
    public:
    using PtNode = shamrock::scheduler::SerialPatchNode<fp_prec_vec>;

    using PatchTree = shamrock::scheduler::PatchTree;

    u32 root_count = 0;
    std::optional<sham::DeviceBuffer<PtNode>> serial_tree_buf;
    std::optional<sham::DeviceBuffer<u64>> linked_patch_ids_buf;

    inline void attach_buf() {
        if (serial_tree_buf.has_value())
            throw shambase::make_except_with_loc<std::runtime_error>(
                "serial_tree_buf is already allocated");
        if (linked_patch_ids_buf.has_value())
            throw shambase::make_except_with_loc<std::runtime_error>(
                "linked_patch_ids_buf is already allocated");

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        serial_tree_buf.emplace(serial_tree.size(), dev_sched);
        serial_tree_buf->copy_from_stdvec(serial_tree);

        linked_patch_ids_buf.emplace(linked_patch_ids.size(), dev_sched);
        linked_patch_ids_buf->copy_from_stdvec(linked_patch_ids);
    }

    inline void detach_buf() {
        if (!serial_tree_buf.has_value())
            throw shambase::make_except_with_loc<std::runtime_error>(
                "serial_tree_buf wasn't allocated");
        if (!linked_patch_ids_buf.has_value())
            throw shambase::make_except_with_loc<std::runtime_error>(
                "linked_patch_ids_buf wasn't allocated");

        serial_tree_buf.reset();
        linked_patch_ids_buf.reset();
    }

    private:
    u32 level_count = 0;

    std::vector<PtNode> serial_tree;
    std::vector<u64> linked_patch_ids;
    std::vector<u64> roots_ids;

    void build_from_patch_tree(
        PatchTree &ptree, const shamrock::patch::PatchCoordTransform<fp_prec_vec> box_transform);

    public:
    inline void print_status() {
        if (shamcomm::world_rank() == 0) {
            for (PtNode n : serial_tree) {
                logger::raw_ln(
                    n.box_min,
                    n.box_max,
                    "[",
                    n.childs_id[0],
                    n.childs_id[1],
                    n.childs_id[2],
                    n.childs_id[3],
                    n.childs_id[4],
                    n.childs_id[5],
                    n.childs_id[6],
                    n.childs_id[7],
                    "]");
            }
        }
    }

    inline SerialPatchTree(
        PatchTree &ptree, const shamrock::patch::PatchCoordTransform<fp_prec_vec> box_transform) {
        StackEntry stack_loc{};
        build_from_patch_tree(ptree, box_transform);
    }

    template<class Acc1, class Acc2>
    inline void host_for_each_leafs_internal(
        std::function<bool(u64, PtNode pnode)> interact_cd,
        std::function<void(u64, PtNode)> found_case,
        Acc1 &&tree,
        Acc2 &&lpid) {

        std::stack<u64> id_stack;

        for (u64 root : roots_ids) {
            id_stack.push(root);
        }

        while (!id_stack.empty()) {
            u64 cur_id = id_stack.top();
            id_stack.pop();
            PtNode cur_p = tree[cur_id];

            bool interact = interact_cd(cur_id, cur_p);

            if (interact) {
                u64 linked_id = lpid[cur_id];
                if (linked_id != u64_max) {
                    found_case(linked_id, cur_p);
                } else {
                    id_stack.push(cur_p.childs_id[0]);
                    id_stack.push(cur_p.childs_id[1]);
                    id_stack.push(cur_p.childs_id[2]);
                    id_stack.push(cur_p.childs_id[3]);
                    id_stack.push(cur_p.childs_id[4]);
                    id_stack.push(cur_p.childs_id[5]);
                    id_stack.push(cur_p.childs_id[6]);
                    id_stack.push(cur_p.childs_id[7]);
                }
            }
        }
    }

    inline void host_for_each_leafs(
        std::function<bool(u64, PtNode pnode)> interact_cd,
        std::function<void(u64, PtNode)> found_case) {
        StackEntry stack_loc{false};

        auto tree = serial_tree_buf->template mirror_to<sham::host>();
        auto lpid = linked_patch_ids_buf->template mirror_to<sham::host>();

        host_for_each_leafs_internal(interact_cd, found_case, tree, lpid);
    }

    /**
     * @brief accesor to the number of level in the tree
     *
     * @return const u32& number of level
     */
    inline const u32 &get_level_count() { return level_count; }

    /**
     * @brief accesor to the number of element in the tree
     *
     * @return const u32& number of element
     */
    inline u32 get_element_count() { return serial_tree.size(); }

    inline static SerialPatchTree<fp_prec_vec> build(PatchScheduler &sched) {
        return SerialPatchTree<fp_prec_vec>(
            sched.patch_tree, sched.get_patch_transform<fp_prec_vec>());
    }

    template<class T, class Func>
    inline shamrock::patch::PatchtreeField<T> make_patch_tree_field(
        PatchScheduler &sched,
        sham::DeviceScheduler_ptr dev_sched,
        shamrock::patch::PatchField<T> pfield,
        Func &&reducer) {
        shamrock::patch::PatchtreeField<T> ptfield;
        ptfield.allocate(get_element_count(), dev_sched);

        {
            auto lpid       = linked_patch_ids_buf->template mirror_to<sham::host>();
            auto tree_field = ptfield.internal_buf->template mirror_to<sham::host>();

            // init reduction
            std::unordered_map<u64, u64> &idp_to_gid = sched.patch_list.id_patch_to_global_idx;
            for (u64 idx = 0; idx < get_element_count(); idx++) {
                tree_field[idx] = (lpid[idx] != u64_max) ? pfield.get(lpid[idx]) : T();
            }
        }

        auto &q       = shambase::get_check_ref(dev_sched).get_queue();
        u32 end_loop  = get_level_count();
        u32 elem_cnt  = get_element_count();

        for (u32 level = 0; level < end_loop; level++) {
            sham::kernel_call(
                q,
                sham::MultiRef{*serial_tree_buf},
                sham::MultiRef{*ptfield.internal_buf},
                elem_cnt,
                [reducer](u32 i, const PtNode *tree, T *f) {
                    std::array<u64, 8> n = tree[i].childs_id;

                    if (n[0] != u64_max) {
                        f[i] = reducer(
                            f[n[0]], f[n[1]], f[n[2]], f[n[3]], f[n[4]], f[n[5]], f[n[6]], f[n[7]]);
                    }
                });
        }
        return ptfield;
    }

    inline void dump_dat() {
        for (u64 idx = 0; idx < get_element_count(); idx++) {
            std::cout << idx << " (" << serial_tree[idx].childs_id[0] << ", "
                      << serial_tree[idx].childs_id[1] << ", " << serial_tree[idx].childs_id[2]
                      << ", " << serial_tree[idx].childs_id[3] << ", "
                      << serial_tree[idx].childs_id[4] << ", " << serial_tree[idx].childs_id[5]
                      << ", " << serial_tree[idx].childs_id[6] << ", "
                      << serial_tree[idx].childs_id[7] << ")";

            std::cout << " (" << serial_tree[idx].box_min.x() << ", "
                      << serial_tree[idx].box_min.y() << ", " << serial_tree[idx].box_min.z()
                      << ")";

            std::cout << " (" << serial_tree[idx].box_max.x() << ", "
                      << serial_tree[idx].box_max.y() << ", " << serial_tree[idx].box_max.z()
                      << ")";

            std::cout << " = " << linked_patch_ids[idx];

            std::cout << std::endl;
        }
    }

    sham::DeviceBuffer<u64> compute_patch_owner(
        sham::DeviceScheduler_ptr dev_sched,
        sham::DeviceBuffer<fp_prec_vec> &position_buffer,
        u32 len);
};

template<class vec>
sham::DeviceBuffer<u64> SerialPatchTree<vec>::compute_patch_owner(
    sham::DeviceScheduler_ptr dev_sched, sham::DeviceBuffer<vec> &position_buffer, u32 len) {
    sham::DeviceBuffer<u64> new_owned_id(len, dev_sched);

    using namespace shamrock::patch;

    sham::DeviceBuffer<u64> roots(roots_ids.size(), dev_sched);
    roots.copy_from_stdvec(roots_ids);

    auto &q       = dev_sched->get_queue();
    u32 root_cnt  = roots_ids.size();
    auto max_lev  = get_level_count();

    using PtNode = shamrock::scheduler::SerialPatchNode<vec>;

    sham::kernel_call(
        q,
        sham::MultiRef{position_buffer, *serial_tree_buf, *linked_patch_ids_buf, roots},
        sham::MultiRef{new_owned_id},
        len,
        [root_cnt, max_lev](
            u32 i,
            const vec *pos,
            const PtNode *tnode,
            const u64 *linked_node_id,
            const u64 *roots_id,
            u64 *new_id) {
            auto xyz = pos[i];

            u64 current_node = 0;

            // find the correct root to start the search
            for (u32 iroot = 0; iroot < root_cnt; iroot++) {
                u32 root_id      = roots_id[iroot];
                PtNode root_node = tnode[root_id];

                if (Patch::is_in_patch_converted(xyz, root_node.box_min, root_node.box_max)) {
                    current_node = root_id;
                    break;
                }
            }

            u64 result_node = u64_max;

            for (u32 step = 0; step < max_lev + 1; step++) {
                PtNode cur_node = tnode[current_node];

                if (cur_node.childs_id[0] != u64_max) {

                    if (Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[0]].box_min,
                            tnode[cur_node.childs_id[0]].box_max)) {
                        current_node = cur_node.childs_id[0];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[1]].box_min,
                            tnode[cur_node.childs_id[1]].box_max)) {
                        current_node = cur_node.childs_id[1];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[2]].box_min,
                            tnode[cur_node.childs_id[2]].box_max)) {
                        current_node = cur_node.childs_id[2];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[3]].box_min,
                            tnode[cur_node.childs_id[3]].box_max)) {
                        current_node = cur_node.childs_id[3];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[4]].box_min,
                            tnode[cur_node.childs_id[4]].box_max)) {
                        current_node = cur_node.childs_id[4];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[5]].box_min,
                            tnode[cur_node.childs_id[5]].box_max)) {
                        current_node = cur_node.childs_id[5];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[6]].box_min,
                            tnode[cur_node.childs_id[6]].box_max)) {
                        current_node = cur_node.childs_id[6];
                    } else if (
                        Patch::is_in_patch_converted(
                            xyz,
                            tnode[cur_node.childs_id[7]].box_min,
                            tnode[cur_node.childs_id[7]].box_max)) {
                        current_node = cur_node.childs_id[7];
                    }

                } else {

                    result_node = linked_node_id[current_node];
                    break;
                }
            }

            new_id[i] = result_node;
        });

    return new_owned_id;
}
