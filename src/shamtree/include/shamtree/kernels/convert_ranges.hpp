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
 * @file convert_ranges.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammath/sfc/morton.hpp"
#include <memory>

template<class u_morton, class vec_prec>
void sycl_convert_cell_range(
    sycl::queue &queue,

    u32 leaf_cnt,
    u32 internal_cnt,

    vec_prec bounding_box_min,
    vec_prec bounding_box_max,

    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>>
        &buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>>
        &buf_pos_max_cell,

    std::unique_ptr<sycl::buffer<vec_prec>> &buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<vec_prec>> &buf_pos_max_cell_flt);
