// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file compute_ranges.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "aliases.hpp"
#include "shamrock/sfc/morton.hpp"
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
        &buf_pos_max_cell
);
