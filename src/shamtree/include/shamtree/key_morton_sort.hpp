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
 * @file key_morton_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include <memory>

enum SortImplType { MultiKernel };

/**
 * @brief sort morton code and generate remap table
 *
 * @tparam u_morton morton precision
 * @param queue sycl queue
 * @param morton_count_rounded_pow morton codes count (equal to a power of 2)
 * @param buf_index buffer countaining the index map corresponding to the sorting
 * @param buf_morton morton buffer that will be sorted
 */
template<class u_morton, SortImplType impl = MultiKernel>
void sycl_sort_morton_key_pair(
    sycl::queue &queue,
    u32 morton_count_rounded_pow,
    std::unique_ptr<sycl::buffer<u32>> &buf_index,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton);
