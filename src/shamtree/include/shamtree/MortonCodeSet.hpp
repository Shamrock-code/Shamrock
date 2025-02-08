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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

        /// The count of Morton codes in the set (rounded to a power of 2)
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
     * @brief Constructs a MortonCodeSet
     *
     * @param dev_sched The device scheduler for managing SYCL operations
     * @param bounding_box The bounding box encapsulating the input positions
     * @param pos_buf The buffer containing the input positions
     * @param cnt_obj The number of positions in the buffer
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSet<Tmorton, Tvec, dim> morton_code_set_from_positions(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::AABB<Tvec> bounding_box,
        sham::DeviceBuffer<Tvec> &pos_buf,
        u32 cnt_obj,
        u32 morton_count);

} // namespace shamtree
