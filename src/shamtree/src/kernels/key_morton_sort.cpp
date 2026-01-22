// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file key_morton_sort.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/key_morton_sort.hpp"
#include "shamalgs/algorithm.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"

template<>
void sycl_sort_morton_key_pair<u32, MultiKernel>(
    sycl::queue &queue,
    u32 morton_count_rounded_pow,
    std::unique_ptr<sycl::buffer<u32>> &buf_index,
    std::unique_ptr<sycl::buffer<u32>> &buf_morton) {

    shamalgs::algorithm::sort_by_key(queue, *buf_morton, *buf_index, morton_count_rounded_pow);
}

template<>
void sycl_sort_morton_key_pair<u64, MultiKernel>(
    sycl::queue &queue,
    u32 morton_count_rounded_pow,
    std::unique_ptr<sycl::buffer<u32>> &buf_index,
    std::unique_ptr<sycl::buffer<u64>> &buf_morton) {

    shamalgs::algorithm::sort_by_key(queue, *buf_morton, *buf_index, morton_count_rounded_pow);
}
