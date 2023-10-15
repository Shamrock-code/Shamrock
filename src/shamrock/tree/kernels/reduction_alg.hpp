// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file reduction_alg.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "aliases.hpp"
#include "flags.hpp"

#include <memory>
#include <vector>
#include "shambase/sycl.hpp"


template<class u_morton>
void reduction_alg(
    //in
    sycl::queue & queue,
    u32 morton_count,
    std::unique_ptr<sycl::buffer<u_morton>> & buf_morton,
    u32 reduction_level,
    //out
    std::unique_ptr<sycl::buffer<u32>> &buf_reduc_index_map,
    u32 & morton_leaf_count);

template<class u_morton>
void sycl_morton_remap_reduction(
    //in
    sycl::queue & queue,
    u32 morton_leaf_count,
    std::unique_ptr<sycl::buffer<u32>> & buf_reduc_index_map,
    std::unique_ptr<sycl::buffer<u_morton>> & buf_morton,
    //out
    std::unique_ptr<sycl::buffer<u_morton>> & buf_leaf_morton);