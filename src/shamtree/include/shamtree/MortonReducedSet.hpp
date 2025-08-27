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
 * @file MortonReducedSet.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/CellIterator.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"

namespace shamtree {

    /**
     * @brief Class representing a set of Morton codes with associated bounding box and position
     * data that was reduced.
     *
     * @tparam Tmorton The type used for Morton codes
     * @tparam Tvec The vector type representing positions
     * @tparam dim The dimensionality, inferred from Tvec if not provided
     */
    template<class Tmorton, class Tvec, u32 dim = shambase::VectorProperties<Tvec>::dimension>
    class MortonReducedSet {
        public:
        /// The source Morton codes set
        MortonCodeSortedSet<Tmorton, Tvec> morton_codes_set;

        /**
         * @brief The count of Morton codes in the reduced set
         * This was called tree_leaf_count
         */
        u32 reduce_code_count;

        /// Indexes of the morton codes in the reduced set
        /// This was called buf_reduc_index_map
        sham::DeviceBuffer<u32> buf_reduc_index_map;

        /// The reduced Morton codes
        /// This was called buf_tree_morton
        sham::DeviceBuffer<Tmorton> reduced_morton_codes;

        /// Move constructor from each members
        MortonReducedSet(
            MortonCodeSortedSet<Tmorton, Tvec> &&morton_codes_set,
            u32 reduce_code_count,
            sham::DeviceBuffer<u32> &&buf_reduc_index_map,
            sham::DeviceBuffer<Tmorton> &&reduced_morton_codes)
            : morton_codes_set(std::move(morton_codes_set)), reduce_code_count(reduce_code_count),
              buf_reduc_index_map(std::move(buf_reduc_index_map)),
              reduced_morton_codes(std::move(reduced_morton_codes)) {}

        inline LeafCellIterator get_cell_iterator() {
            return LeafCellIterator{morton_codes_set.map_morton_id_to_obj_id, buf_reduc_index_map};
        }

        inline LeafCellIteratorHost get_cell_iterator_host() const {
            return LeafCellIteratorHost{
                morton_codes_set.map_morton_id_to_obj_id.copy_to_stdvec(),
                buf_reduc_index_map.copy_to_stdvec()};
        }

        inline static MortonReducedSet make_empty(sham::DeviceScheduler_ptr dev_sched) {
            return MortonReducedSet{
                MortonCodeSortedSet<Tmorton, Tvec>::make_empty(dev_sched),
                0,
                sham::DeviceBuffer<u32>(0, dev_sched),
                sham::DeviceBuffer<Tmorton>(0, dev_sched)};
        }
    };

    /**
     * @brief Reduces the given Morton code set by grouping together Morton codes
     * that are close to each other in the tree.
     *
     * @param dev_sched The device scheduler to use for the reduction
     * @param morton_codes_set The set of Morton codes to reduce
     * @param reduction_level The amount of reduction to apply
     * @return The reduced set of Morton codes
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonReducedSet<Tmorton, Tvec, dim> reduce_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSortedSet<Tmorton, Tvec, dim> &&morton_codes_set,
        u32 reduction_level);

    /**
     * @brief Reduces the given Morton code set by grouping together Morton codes
     * that are close to each other in the tree.
     *
     * @param dev_sched The device scheduler to use for the reduction
     * @param morton_codes_set The set of Morton codes to reduce
     * @param reduction_level The amount of reduction to apply
     * @param cache_buf_reduc_index_map A device buffer to be reused for the reduction index map
     * @param cache_reduced_morton_codes A device buffer to be reused for the reduced Morton codes
     * @return The reduced set of Morton codes
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonReducedSet<Tmorton, Tvec, dim> reduce_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSortedSet<Tmorton, Tvec, dim> &&morton_codes_set,
        u32 reduction_level,
        sham::DeviceBuffer<u32> &&cache_buf_reduc_index_map,
        sham::DeviceBuffer<Tmorton> &&cache_reduced_morton_codes);

} // namespace shamtree
