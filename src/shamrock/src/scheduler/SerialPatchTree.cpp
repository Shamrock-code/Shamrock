// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SerialPatchTree.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/log.hpp"

using PatchTree = shamrock::scheduler::PatchTree;

template<class vec>
void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<vec>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<vec> box_transform) {

    PatchTree::Node ptn = ptree.tree[root_key];

    using PtNode = shamrock::scheduler::SerialPatchNode<vec>;
    PtNode n     = ptn.convert(box_transform);

    max_level = sycl::max((u32) ptn.tree_node.level, max_level);

    u64 id_nvec = counter;
    result_tree.push_back(n);
    result_tree_linked_patch_id.push_back(ptn.linked_patchid);
    counter++;

    // std::cout << "id : " << counter-1 << " leaf : " << ptn.is_leaf << std::endl;

    if (!ptn.is_leaf()) {

        u64 old_id0 = ptn.get_child_nid(0);
        u64 old_id1 = ptn.get_child_nid(1);
        u64 old_id2 = ptn.get_child_nid(2);
        u64 old_id3 = ptn.get_child_nid(3);
        u64 old_id4 = ptn.get_child_nid(4);
        u64 old_id5 = ptn.get_child_nid(5);
        u64 old_id6 = ptn.get_child_nid(6);
        u64 old_id7 = ptn.get_child_nid(7);

        result_tree.at(id_nvec).childs_id[0] = counter;
        get_serial_tree<vec>(
            old_id0,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[1] = counter;
        get_serial_tree<vec>(
            old_id1,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[2] = counter;
        get_serial_tree<vec>(
            old_id2,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[3] = counter;
        get_serial_tree<vec>(
            old_id3,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[4] = counter;
        get_serial_tree<vec>(
            old_id4,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[5] = counter;
        get_serial_tree<vec>(
            old_id5,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[6] = counter;
        get_serial_tree<vec>(
            old_id6,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);

        result_tree.at(id_nvec).childs_id[7] = counter;
        get_serial_tree<vec>(
            old_id7,
            ptree,
            result_tree,
            result_tree_linked_patch_id,
            counter,
            max_level,
            box_transform);
    }
}

template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<f32_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<f64_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<f64_3> box_transform);

template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<i64_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<i64_3> box_transform);

template<>
void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        shamlog_debug_ln("Serial Patch Tree", "get serial tree fp32 root id :", root_id);
        get_serial_tree<f32_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template<>
void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        get_serial_tree<f64_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }
    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template<>
void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        get_serial_tree<i64_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }
    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}
