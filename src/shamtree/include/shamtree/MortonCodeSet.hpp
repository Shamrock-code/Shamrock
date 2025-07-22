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
 * @file MortonCodeSet.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"

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
    class MortonCodeSet {
        public:
        /// The axis-aligned bounding box for the set of positions
        shammath::AABB<Tvec> bounding_box;

        /// The count of objects represented in the set
        u32 cnt_obj;

        /// The count of Morton codes in the set (every code after cnt_obj is err_code)
        u32 morton_count;

        /// Device buffer holding the Morton codes
        sham::DeviceBuffer<Tmorton> morton_codes;

        /// Move constructor from each members
        MortonCodeSet(
            shammath::AABB<Tvec> &&bounding_box,
            u32 &&cnt_obj,
            u32 &&morton_count,
            sham::DeviceBuffer<Tmorton> &&morton_codes)
            : bounding_box(std::move(bounding_box)), cnt_obj(std::move(cnt_obj)),
              morton_count(std::move(morton_count)), morton_codes(std::move(morton_codes)) {}
    };

    /**
     * @brief Generate a set of Morton codes from a buffer of positions
     *
     * @param dev_sched The device scheduler for the computation
     * @param bounding_box The bounding box containing all positions
     * @param pos_buf The device buffer containing the positions
     * @param cnt_obj The count of objects in the buffer
     * @param morton_count The count of Morton codes in the output set
     * (can be different from cnt_obj)
     * @param cache_buf_morton_codes A device buffer to be reused for the output Morton codes
     *
     * If morton_count > cnt_obj, the extra Morton codes will be set to an error code larger than
     * any valid Morton code.
     *
     * @return The MortonCodeSet, containing the bounding box, the count of objects
     * and the Morton codes. The Morton codes are sorted in ascending order.
     *
     * @note morton_count >= cnt_obj
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSet<Tmorton, Tvec, dim> morton_code_set_from_positions(
        const sham::DeviceScheduler_ptr &dev_sched,
        shammath::AABB<Tvec> bounding_box,
        sham::DeviceBuffer<Tvec> &pos_buf,
        u32 cnt_obj,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &&cache_buf_morton_codes);

    /**
     * @brief Generate a set of Morton codes from a buffer of positions
     *
     * @param dev_sched The device scheduler for the computation
     * @param bounding_box The bounding box containing all positions
     * @param pos_buf The device buffer containing the positions
     * @param cnt_obj The count of objects in the buffer
     * @param morton_count The count of Morton codes in the output set
     * (can be different from cnt_obj)
     *
     * If morton_count > cnt_obj, the extra Morton codes will be set to an error code larger than
     * any valid Morton code.
     *
     * @return The MortonCodeSet, containing the bounding box, the count of objects
     * and the Morton codes. The Morton codes are sorted in ascending order.
     *
     * @note morton_count >= cnt_obj
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSet<Tmorton, Tvec, dim> morton_code_set_from_positions(
        const sham::DeviceScheduler_ptr &dev_sched,
        shammath::AABB<Tvec> bounding_box,
        sham::DeviceBuffer<Tvec> &pos_buf,
        u32 cnt_obj,
        u32 morton_count);

} // namespace shamtree
