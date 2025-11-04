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
 * @file MortonCodeSortedSet.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/MortonCodeSet.hpp"

namespace shamtree {

    /**
     * @brief Class representing a set of Morton codes with associated bounding box and position
     * data
     *
     * @tparam Tmorton The type used for Morton codes
     * @tparam Tvec The vector type representing positions
     * @tparam dim The dimensionality, inferred from Tvec if not provided
     */
    template<class Tmorton, class Tvec, u32 dim = shambase::VectorProperties<Tvec>::dimension>
    class MortonCodeSortedSet {
        public:
        /// The axis-aligned bounding box for the set of positions
        shammath::AABB<Tvec> bounding_box;

        /// The count of objects represented in the set
        u32 cnt_obj;

        /// The count of Morton codes in the set (every code after cnt_obj is err_code)
        u32 morton_count;

        /// Device buffer holding the sorted Morton codes
        sham::DeviceBuffer<Tmorton> sorted_morton_codes;

        /// Device buffer holding the map from sorted Morton code to object id
        sham::DeviceBuffer<u32> map_morton_id_to_obj_id;

        /// Move constructor from each members
        MortonCodeSortedSet(
            shammath::AABB<Tvec> &&bounding_box,
            u32 &&cnt_obj,
            u32 &&morton_count,
            sham::DeviceBuffer<Tmorton> &&sorted_morton_codes,
            sham::DeviceBuffer<u32> &&map_morton_id_to_obj_id)
            : bounding_box(std::move(bounding_box)), cnt_obj(std::move(cnt_obj)),
              morton_count(std::move(morton_count)),
              sorted_morton_codes(std::move(sorted_morton_codes)),
              map_morton_id_to_obj_id(std::move(map_morton_id_to_obj_id)) {}

        inline static MortonCodeSortedSet make_empty(sham::DeviceScheduler_ptr dev_sched) {
            return MortonCodeSortedSet(
                shammath::AABB<Tvec>(),
                0_u32,
                0_u32,
                sham::DeviceBuffer<Tmorton>(0, dev_sched),
                sham::DeviceBuffer<u32>(0, dev_sched));
        }
    };

    /**
     * @brief Sorts a set of Morton codes and creates a new
     * MortonCodeSortedSet object with the sorted codes and the
     * associated map from sorted Morton code to object id
     *
     * @param dev_sched The SYCL device scheduler to use
     * @param morton_codes_set The MortonCodeSet to be sorted
     * @return A new MortonCodeSortedSet object with the sorted codes and the associated map
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set);

    /**
     * @brief Sorts a set of Morton codes and creates a new
     * MortonCodeSortedSet object with the sorted codes and the
     * associated map from sorted Morton code to object id
     *
     * @param dev_sched The SYCL device scheduler to use
     * @param morton_codes_set The MortonCodeSet to be sorted
     * @param cached_map_morton_id_to_obj_id A pre-allocated device buffer to store the map
     * from sorted Morton code to object id
     * @return A new MortonCodeSortedSet object with the sorted codes and the associated map
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set,
        sham::DeviceBuffer<u32> &&cached_map_morton_id_to_obj_id);

} // namespace shamtree
