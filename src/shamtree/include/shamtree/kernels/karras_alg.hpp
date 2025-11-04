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
 * @file karras_alg.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include <memory>

/**
 * @brief Karras 2012 algorithm with addition endrange buffer
 *
 * Given a list of morton codes, compute the left and right child id, left and right
 * child flag, and the endrange for each cell using the Karras 2012 algorithm.
 *
 * @param[in] queue sycl queue
 * @param[in] internal_cell_count number of internal cells
 * @param[in] in_morton input morton codes
 * @param[out] out_buf_lchild_id left child id
 * @param[out] out_buf_rchild_id right child id
 * @param[out] out_buf_lchild_flag left child flag
 * @param[out] out_buf_rchild_flag right child flag
 * @param[out] out_buf_endrange endrange
 */
template<class u_morton>
void sycl_karras_alg(
    sycl::queue &queue,
    u32 internal_cell_count,
    sycl::buffer<u_morton> &in_morton,
    sycl::buffer<u32> &out_buf_lchild_id,
    sycl::buffer<u32> &out_buf_rchild_id,
    sycl::buffer<u8> &out_buf_lchild_flag,
    sycl::buffer<u8> &out_buf_rchild_flag,
    sycl::buffer<u32> &out_buf_endrange);
