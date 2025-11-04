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
 * @file CompressedLeafBVH.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammath/sfc/morton.hpp"
#include "shamtree/CLBVHObjectIterator.hpp"
#include "shamtree/CellIterator.hpp"
#include "shamtree/KarrasRadixTree.hpp"
#include "shamtree/KarrasRadixTreeAABB.hpp"
#include "shamtree/MortonReducedSet.hpp"

namespace shamtree {

    /**
     * @class CompressedLeafBVH
     * @brief A Compressed Leaf Bounding Volume Hierarchy (CLBVH) for neighborhood queries.
     *
     * A Compressed Leaf Bounding Volume Hierarchy (CLBVH) is a hierarchical data structure used for
     * efficient collision detection. It is composed of a MortonReducedSet to store the reduced set
     * of Morton codes (Compressed aspect of the tree), a KarrasRadixTree to represent the tree
     * structure, and a KarrasRadixTreeAABB to store the bounding boxes of the tree cells.
     *
     * @tparam Tmorton The type used to represent the Morton codes.
     * @tparam Tvec The type used to represent the 3D vector positions.
     * @tparam dim The number of dimensions (3 for 3D).
     */
    template<class Tmorton, class Tvec, u32 dim>
    class CompressedLeafBVH;

} // namespace shamtree

template<class Tmorton, class Tvec, u32 dim>
class shamtree::CompressedLeafBVH {
    public:
    /// Get internal cell count
    inline u32 get_total_cell_count() { return structure.get_total_cell_count(); }

    /// Get internal cell count
    inline u32 get_internal_cell_count() { return structure.get_internal_cell_count(); }

    /// is the root a leaf ?
    inline bool is_root_leaf() const { return structure.is_root_leaf(); }

    /// Get leaf cell count
    inline u32 get_leaf_cell_count() { return structure.get_leaf_count(); }

    /// The reduced set of Morton codes
    MortonReducedSet<Tmorton, Tvec, dim> reduced_morton_set;

    /// The tree structure
    KarrasRadixTree structure;

    /// The bounding box of the tree cells
    KarrasRadixTreeAABB<Tvec> aabbs;

    /**
     * @brief Construct a new CompressedLeafBVH from a MortonReducedSet,
     *        a KarrasRadixTree, and a KarrasRadixTreeAABB.
     *
     * @param[in] reduced_morton_set the MortonReducedSet to take ownership of
     * @param[in] structure the KarrasRadixTree to take ownership of
     * @param[in] aabbs the KarrasRadixTreeAABB to take ownership of
     */
    CompressedLeafBVH(
        MortonReducedSet<Tmorton, Tvec, dim> &&reduced_morton_set,
        KarrasRadixTree &&structure,
        KarrasRadixTreeAABB<Tvec> &&aabbs)
        : reduced_morton_set(
              std::forward<MortonReducedSet<Tmorton, Tvec, dim>>(reduced_morton_set)),
          structure(std::forward<KarrasRadixTree>(structure)),
          aabbs(std::forward<KarrasRadixTreeAABB<Tvec>>(aabbs)) {}

    /// make an empty BVH
    static CompressedLeafBVH make_empty(sham::DeviceScheduler_ptr dev_sched);

    /// is the BVH empty ?
    inline bool is_empty() const { return reduced_morton_set.is_empty(); }

    /**
     * @brief rebuild the BVH from the given positions
     *
     * @param[in] positions the positions of the particles
     * @param[in] bounding_box the bounding box of the particles
     * @param[in] compression_level the compression level of the BVH
     */
    void rebuild_from_positions(
        sham::DeviceBuffer<Tvec> &positions,
        const shammath::AABB<Tvec> &bounding_box,
        u32 compression_level);

    /**
     * @brief rebuild the BVH from the given positions
     *
     * @param[in] positions the positions of the particles
     * @param[in] obj_cnt the number of particles
     * @param[in] bounding_box the bounding box of the particles
     * @param[in] compression_level the compression level of the BVH
     */
    void rebuild_from_positions(
        sham::DeviceBuffer<Tvec> &positions,
        u32 obj_cnt,
        const shammath::AABB<Tvec> &bounding_box,
        u32 compression_level);

#if false
    void rebuild_from_position_range(
        sham::DeviceBuffer<Tvec> &min,
        sham::DeviceBuffer<Tvec> &max,
        shammath::AABB<Tvec> &bounding_box,
        u32 compression_level);
#endif

    inline shamtree::CLBVHTraverser<Tmorton, Tvec, dim> get_traverser() const {
        return {structure.get_structure_traverser(), aabbs.buf_aabb_min, aabbs.buf_aabb_max};
    }

    inline shamtree::CLBVHTraverserHost<Tmorton, Tvec, dim> get_traverser_host() const {
        return {
            structure.get_structure_traverser_host(),
            aabbs.buf_aabb_min.copy_to_stdvec(),
            aabbs.buf_aabb_max.copy_to_stdvec()};
    }

    /**
     * @brief Retrieves an iterator for traversing objects in the BVH.
     *
     * This function returns a CLBVHObjectIterator configured to traverse
     * the objects in the compressed leaf BVH. The iterator is initialized
     * using the cell iterator from the reduced Morton set, the structure
     * traverser from the Karras Radix Tree structure, and the minimum and
     * maximum AABB buffers.
     *
     * @return A CLBVHObjectIterator for object traversal.
     */
    inline shamtree::CLBVHObjectIterator<Tmorton, Tvec, dim> get_object_iterator() const {
        return {reduced_morton_set.get_leaf_cell_iterator(), get_traverser()};
    }

    inline shamtree::CLBVHObjectIteratorHost<Tmorton, Tvec, dim> get_object_iterator_host() const {
        return {reduced_morton_set.get_leaf_cell_iterator_host(), get_traverser_host()};
    }

    inline CellIterator get_cell_iterator() const {
        return {reduced_morton_set.get_cell_iterator(
            structure.buf_endrange, structure.get_internal_cell_count())};
    }

    inline CellIteratorHost get_cell_iterator_host() const {
        return {reduced_morton_set.get_cell_iterator_host(
            structure.buf_endrange, structure.get_internal_cell_count())};
    }
};
