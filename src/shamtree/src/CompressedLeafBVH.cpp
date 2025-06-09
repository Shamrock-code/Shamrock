// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CompressedLeafBVH.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

template<class Tmorton, class Tvec, u32 dim>
shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>
shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::make_empty(sham::DeviceScheduler_ptr dev_sched) {
    StackEntry stack_loc{};
    return {
        MortonReducedSet<Tmorton, Tvec, dim>::make_empty(dev_sched),
        KarrasRadixTree::make_empty(dev_sched),
        KarrasRadixTreeAABB<Tvec>::make_empty(dev_sched)};
}

template<class Tmorton, class Tvec, u32 dim>
void shamtree::CompressedLeafBVH<Tmorton, Tvec, dim>::rebuild_from_positions(
    sham::DeviceBuffer<Tvec> &positions,
    shammath::AABB<Tvec> &bounding_box,
    u32 compression_level) {
    StackEntry stack_loc{};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    u32 roundup_pow2 = shambase::roundup_pow2(positions.get_size());

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, dim>(
        dev_sched,
        bounding_box,
        positions,
        positions.get_size(),
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

    auto tree_aabbs = shamtree::compute_tree_aabb_from_positions(
        tree, reduced_set.get_cell_iterator(), std::move(aabbs), positions);

    this->reduced_morton_set = std::move(reduced_set);
    this->structure          = std::move(tree);
    this->aabbs              = std::move(tree_aabbs);
}

template class shamtree::CompressedLeafBVH<u32, f64_3, 3>;
template class shamtree::CompressedLeafBVH<u64, f64_3, 3>;
