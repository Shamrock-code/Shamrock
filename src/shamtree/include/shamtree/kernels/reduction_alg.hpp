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
 * @file reduction_alg.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"
#include <memory>
#include <vector>

/**
 * @brief Reduces a Morton tree on device.
 *
 * This function will reduce a Morton tree represented by a device buffer
 * buf_morton containing morton_count elements. The reduction is done using
 * the following steps:
 *  - generate a split table for the given tree
 *  - perform a reduction on the tree using the split table
 *  - compute the reduction index map
 *  - remap the tree Morton codes using the reduction index map
 *
 * The function takes a device scheduler dev_sched as input and will use it to
 * schedule the kernel calls.
 *
 * The function returns a struct of two elements:
 *  - buf_reduc_index_map: a device buffer containing the reduction index map
 *  - morton_leaf_count: the number of leaf nodes in the reduced tree
 *
 * @param queue: a SYCL queue to submit the kernels
 * @param morton_count: the number of elements in the input Morton tree
 * @param buf_morton: a device buffer containing the Morton tree
 * @param reduction_level: the number of reduction levels to perform
 * @param buf_reduc_index_map: a device buffer containing the reduction index map
 * @param morton_leaf_count: the number of leaf nodes in the reduced tree
 */
template<class u_morton>
void reduction_alg(
    // in
    sycl::queue &queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    u32 reduction_level,
    // out
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    u32 &morton_leaf_count);

/**
 * @brief Remaps a Morton tree on device using a reduction index map.
 *
 * This function takes a device buffer buf_reduc_index_map containing a reduction
 * index map and a device buffer buf_morton containing a Morton tree and applies
 * the reduction index map to the Morton tree. The output is stored in a new
 * device buffer buf_leaf_morton containing the remapped Morton tree.
 *
 * The function takes a device scheduler dev_sched as input and will use it to
 * schedule the kernel calls.
 *
 * @param queue: a SYCL queue to submit the kernels
 * @param morton_leaf_count: the number of leaf nodes in the reduced tree
 * @param buf_reduc_index_map: a device buffer containing the reduction index map
 * @param buf_morton: a device buffer containing the Morton tree
 * @param buf_leaf_morton: a device buffer containing the remapped Morton tree
 */
template<class u_morton>
void sycl_morton_remap_reduction(
    // in
    sycl::queue &queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    // out
    std::unique_ptr<sycl::buffer<u_morton>> &buf_leaf_morton);

/**
 * @brief Return type of reduction algorithms
 *
 * A reduction algorithm return a struct of two elements:
 *  - buf_reduc_index_map: a device buffer containing the reduction index map
 *  - morton_leaf_count: the number of leaf nodes in the reduced tree
 *
 * @tparam split_int: the type of integer used to represent the split table
 */
template<class split_int>
struct reduc_ret_t {
    sham::DeviceBuffer<split_int> buf_reduc_index_map;
    u32 morton_leaf_count;
};

/**
 * @brief Reduces a Morton tree on device.
 *
 * This function will reduce a Morton tree represented by a device buffer
 * buf_morton containing morton_count elements. The reduction is done using
 * the following steps:
 *  - generate a split table for the given tree
 *  - perform a reduction on the tree using the split table
 *  - compute the reduction index map
 *  - remap the tree Morton codes using the reduction index map (using sycl_morton_remap_reduction)
 *
 * The function takes a device scheduler dev_sched as input and will use it to
 * schedule the kernel calls.
 *
 * The function returns a struct of two elements:
 *  - buf_reduc_index_map: a device buffer containing the reduction index map
 *  - morton_leaf_count: the number of leaf nodes in the reduced tree
 *
 * @param dev_sched: a device scheduler to schedule the kernel calls
 * @param morton_count: the number of elements in the input Morton tree
 * @param buf_morton: a device buffer containing the Morton tree
 * @param reduction_level: the number of reduction levels to perform
 *
 * @return a struct containing the reduction index map and the number of leaf
 * nodes in the reduced tree
 */
template<class u_morton>
reduc_ret_t<u32> reduction_alg(
    const sham::DeviceScheduler_ptr &dev_sched,
    u32 morton_count,
    sham::DeviceBuffer<u_morton> &buf_morton,
    u32 reduction_level);

/**
 * @brief Remaps a Morton tree on device using a reduction index map.
 *
 * This function takes a device buffer buf_reduc_index_map containing the
 * reduction index map and a device buffer buf_morton containing the Morton
 * tree. It will remap the Morton tree using the reduction index map and write
 * the result to a device buffer buf_leaf_morton.
 *
 * The function takes a device scheduler queue as input and will use it to
 * schedule the kernel calls.
 *
 * @param queue: a device scheduler to schedule the kernel calls
 * @param morton_leaf_count: the number of leaf nodes in the reduced tree
 * @param buf_reduc_index_map: a device buffer containing the reduction index map
 * @param buf_morton: a device buffer containing the Morton tree
 * @param buf_leaf_morton: a device buffer that will contain the remapped Morton tree
 */
template<class u_morton>
void sycl_morton_remap_reduction(
    // in
    sham::DeviceQueue &queue,
    u32 morton_leaf_count,
    sham::DeviceBuffer<u32> &buf_reduc_index_map,
    sham::DeviceBuffer<u_morton> &buf_morton,
    // out
    sham::DeviceBuffer<u_morton> &buf_leaf_morton);
