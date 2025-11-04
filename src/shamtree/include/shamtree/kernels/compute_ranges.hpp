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
 * @file compute_ranges.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammath/sfc/morton.hpp"
#include <memory>

template<class u_morton>
void sycl_compute_cell_ranges(

    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,
    std::unique_ptr<sycl::buffer<u_morton>> &buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> &buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>> &buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>> &buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> &buf_endrange,
    std::unique_ptr<sycl::buffer<typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr>>
        &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename shamrock::sfc::MortonCodes<u_morton, 3>::int_vec_repr>>
        &buf_pos_max_cell);
