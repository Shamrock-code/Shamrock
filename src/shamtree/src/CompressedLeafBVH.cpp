// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CompressedLeafBVH.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

template<class Tmorton, class Tvec, u32 dim>
shamtree::CompressedLeafBVH<Tmorton, Tvec, dim> shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::
    make_empty(sham::DeviceScheduler_ptr dev_sched) {
    StackEntry stack_loc{};
    return {
        MortonReducedSet<Tmorton, Tvec, dim>::make_empty(dev_sched),
        KarrasRadixTree::make_empty(dev_sched),
        KarrasRadixTreeAABB<Tvec>::make_empty(dev_sched)};
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::internal_rebuild_from_positions_no_aabb(
    sham::DeviceBuffer<Tvec> &positions,
    u32 obj_cnt,
    const shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {
    __shamrock_stack_entry();

    if (obj_cnt == 0) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "obj_cnt is 0, cannot build a CompressedLeafBVH");
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    u32 roundup_pow2 = shambase::roundup_pow2(obj_cnt);

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, dim>(
        dev_sched,
        bounding_box,
        positions,
        obj_cnt,
        roundup_pow2,
        std::move(reduced_morton_set.morton_codes_set.sorted_morton_codes));

    auto sorted_set = shamtree::sort_morton_set(
        dev_sched,
        std::move(set),
        std::move(reduced_morton_set.morton_codes_set.map_morton_id_to_obj_id));

    auto reduced_set = shamtree::reduce_morton_set(
        dev_sched,
        std::move(sorted_set),
        compression_level,
        std::move(reduced_morton_set.buf_reduc_index_map),
        std::move(reduced_morton_set.reduced_morton_codes));

    auto tree = shamtree::karras_tree_from_morton_set(
        dev_sched,
        reduced_set.reduced_morton_codes.get_size(),
        reduced_set.reduced_morton_codes,
        std::move(structure));

    this->reduced_morton_set = std::move(reduced_set);
    this->structure          = std::move(tree);
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::rebuild_from_positions(
    sham::DeviceBuffer<Tvec> &positions,
    u32 obj_cnt,
    const shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {

    this->internal_rebuild_from_positions_no_aabb(
        positions, obj_cnt, bounding_box, compression_level);

    auto tree_aabbs = shamtree::compute_tree_aabb_from_positions(
        this->structure,
        this->reduced_morton_set.get_leaf_cell_iterator(),
        std::move(this->aabbs),
        positions);

    this->aabbs = std::move(tree_aabbs);
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::rebuild_from_position_range(
    sham::DeviceBuffer<Tvec> &min,
    sham::DeviceBuffer<Tvec> &max,
    u32 obj_cnt,
    shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {

    this->internal_rebuild_from_positions_no_aabb(min, obj_cnt, bounding_box, compression_level);

    auto tree_aabbs = shamtree::compute_tree_aabb_from_position_ranges(
        this->structure,
        this->reduced_morton_set.get_leaf_cell_iterator(),
        std::move(this->aabbs),
        min,
        max);

    this->aabbs = std::move(tree_aabbs);
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::rebuild_from_positions(
    sham::DeviceBuffer<Tvec> &positions,
    const shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {
    this->rebuild_from_positions(positions, positions.get_size(), bounding_box, compression_level);
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::rebuild_from_position_range(
    sham::DeviceBuffer<Tvec> &min,
    sham::DeviceBuffer<Tvec> &max,
    shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {
    if (min.get_size() != max.get_size()) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "min and max must have the same size");
    }
    this->rebuild_from_position_range(min, max, min.get_size(), bounding_box, compression_level);
}

template<class Tmorton, class Tvec, u32 dim>
u32 shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::get_exact_tree_depth() const {
    __shamrock_stack_entry();

    // a single leaf root (or an empty tree) has no edge to walk
    if (is_empty() || structure.is_root_leaf()) {
        return 0;
    }

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    u32 int_cell_count = structure.get_internal_cell_count();

    // height of every cell (internal cells first then leaves, leaves are at height 0)
    sham::DeviceBuffer<u32> height(structure.get_total_cell_count(), dev_sched);
    height.fill(0);

    auto traverser = structure.get_structure_traverser();

    // Heights only ever grow towards their exact value, so racing on the in-place updates within
    // a pass is harmless. Each pass propagates the heights up by at least one level, so after
    // tree_depth (an upper bound of the real depth) passes the exact height has reached the root.
    // No convergence check is done to avoid a device to host read back after every pass.
    for (u32 i = 0; i < structure.tree_depth; i++) {
        sham::kernel_call(
            q,
            sham::MultiRef{traverser},
            sham::MultiRef{height},
            int_cell_count,
            [](u32 gid, auto tree_traverser, u32 *height) {
                u32 hl = height[tree_traverser.get_left_child(gid)];
                u32 hr = height[tree_traverser.get_right_child(gid)];

                height[gid] = 1 + sycl::max(hl, hr);
            });
    }

    // the root of a Karras tree is always the cell 0
    return height.get_val_at_idx(0);
}

template class shamtree::CompressedLeafBVH<u32, f64_3, 3>;
template class shamtree::CompressedLeafBVH<u64, f64_3, 3>;
template class shamtree::CompressedLeafBVH<u64, i64_3, 3>;
