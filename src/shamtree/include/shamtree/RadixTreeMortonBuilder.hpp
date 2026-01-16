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
 * @file RadixTreeMortonBuilder.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utility to build morton codes for the radix tree
 * @date 2023-08-02
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"

/**
 * @brief Helper class to build morton codes
 * \todo use Tvec Tmorton and infer dimension from it
 * @tparam morton_t the morton type used
 * @tparam pos_t the position type (vector)
 * @tparam dim dimensionality
 */
template<class morton_t, class pos_t, u32 dim>
class RadixTreeMortonBuilder {
    public:
    /**
     * @brief build morton code table for the tree
     *
     * index map detail : (the given index correspong to the one of an original position)
     *
     * @param queue the SYCL queue to run on
     * @param bounding_box bounding box of the corresponding input coordinates
     * @param pos_buf buffer of position to build morton codes from
     * @param cnt_obj number of position given in the buffer
     * @param out_buf_morton resulting morton buffer (sorted in morton ordering)
     * @param out_buf_particle_index_map resulting index map
     */
    static void build(
        sycl::queue &queue,
        std::tuple<pos_t, pos_t> bounding_box,
        sycl::buffer<pos_t> &pos_buf,
        u32 cnt_obj,
        std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton,
        std::unique_ptr<sycl::buffer<u32>> &out_buf_particle_index_map);

    static void build(
        sham::DeviceScheduler_ptr dev_sched,
        std::tuple<pos_t, pos_t> bounding_box,
        sham::DeviceBuffer<pos_t> &pos_buf,
        u32 cnt_obj,
        std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton,
        std::unique_ptr<sycl::buffer<u32>> &out_buf_particle_index_map);

    /**
     * @brief build a raw mrton table from a position buffer (no sorting & index map)
     *
     * @param queue the SYCL queue to run on
     * @param bounding_box bounding box of the corresponding input coordinates
     * @param pos_buf buffer of position to build morton codes from
     * @param cnt_obj number of position given in the buffer
     * @param out_buf_morton resulting morton buffer (unsorted)
     */
    static void build_raw(
        sycl::queue &queue,
        std::tuple<pos_t, pos_t> bounding_box,
        sycl::buffer<pos_t> &pos_buf,
        u32 cnt_obj,
        std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton);
};
