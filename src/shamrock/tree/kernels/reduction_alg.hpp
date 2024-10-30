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
 * @file reduction_alg.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include <memory>
#include <vector>

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

template<class u_morton>
void sycl_morton_remap_reduction(
    // in
    sycl::queue &queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    // out
    std::unique_ptr<sycl::buffer<u_morton>> &buf_leaf_morton);
