// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file LeafCompressedBVH.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shammath/sfc/morton.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/KarrasRadixTreeAABB.hpp"
#include "shamtree/LCBVHObjectIterator.hpp"
#include "shamtree/MortonReducedSet.hpp"

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    class LeafCompressedBVH;

} // namespace shamtree

template<class Tmorton, class Tvec, u32 dim>
class shamtree::LeafCompressedBVH {
    public:
    MortonReducedSet<Tmorton, Tvec, dim> reduced_morton_set;
    KarrasRadixTree structure;
    KarrasRadixTreeAABB<Tvec> aabbs;

    static LeafCompressedBVH make_empty(sham::DeviceScheduler_ptr dev_sched);

    void rebuild_from_positions(
        sham::DeviceBuffer<Tvec> &positions,
        shammath::AABB<Tvec> &bounding_box,
        u32 compression_level);

    // void rebuild_from_position_range(sham::DeviceBuffer<Tvec> &min, sham::DeviceBuffer<Tvec>
    // &max,
    //  shammath::AABB<Tvec> &bounding_box,
    //  u32 compression_level);

    inline shamtree::LCBVHObjectIterator<Tmorton, Tvec, dim> get_object_iterator() {
        return {
            reduced_morton_set.get_cell_iterator(),
            structure.get_structure_traverser(),
            aabbs.buf_aabb_min,
            aabbs.buf_aabb_max};
    }
};
