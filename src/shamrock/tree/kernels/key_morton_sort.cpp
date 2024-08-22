// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file key_morton_sort.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "key_morton_sort.hpp"
#include "shamalgs/algorithm.hpp"
#include "shamsys/legacy/log.hpp"

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
