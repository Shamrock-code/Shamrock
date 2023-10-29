// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file convert_ranges.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "aliases.hpp"
#include "shamrock/sfc/morton.hpp"
#include <memory>


template<class u_morton, class vec_prec>
void sycl_convert_cell_range(sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt ,

    vec_prec bounding_box_min,
    vec_prec bounding_box_max,

    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_max_cell,
    
    std::unique_ptr<sycl::buffer<vec_prec>> & buf_pos_min_cell_flt,
    std::unique_ptr<sycl::buffer<vec_prec>> & buf_pos_max_cell_flt);
