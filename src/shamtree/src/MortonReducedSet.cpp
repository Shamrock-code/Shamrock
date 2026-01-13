// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonReducedSet.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/MortonReducedSet.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamtree/kernels/reduction_alg.hpp"
#include <utility>

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    MortonReducedSet<Tmorton, Tvec, dim> reduce_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSortedSet<Tmorton, Tvec, dim> &&morton_codes_set,
        u32 reduction_level,
        sham::DeviceBuffer<u32> &&cache_buf_reduc_index_map,
        sham::DeviceBuffer<Tmorton> &&cache_reduced_morton_codes) {

        // cache_buf_reduc_index_map is not yet recycled

        reduc_ret_t<u32> res = reduction_alg(
            dev_sched,
            morton_codes_set.cnt_obj,
            morton_codes_set.sorted_morton_codes,
            reduction_level);

        shamlog_debug_sycl_ln(
            "RadixTree",
            "reduction results : (before :",
            morton_codes_set.cnt_obj,
            " | after :",
            res.morton_leaf_count,
            ") ratio :",
            shambase::format_printf(
                "%2.2f", f32(morton_codes_set.cnt_obj) / f32(res.morton_leaf_count)));

        if (res.morton_leaf_count == 0) {
            throw shambase::make_except_with_loc<std::runtime_error>("0 leaf tree cannot exists");
        }

        // here the old "One cell mode" is not implemented as I want to get rid of this confusing
        // mess, seriously this thing was giving me never ending headaches ...

        sham::DeviceBuffer<Tmorton> buf_tree_morton
            = std::forward<sham::DeviceBuffer<Tmorton>>(cache_reduced_morton_codes);
        buf_tree_morton.resize(res.morton_leaf_count);

        sycl_morton_remap_reduction(
            dev_sched->get_queue(),
            res.morton_leaf_count,
            res.buf_reduc_index_map,
            morton_codes_set.sorted_morton_codes,
            buf_tree_morton);

        return MortonReducedSet<Tmorton, Tvec, dim>(
            std::forward<MortonCodeSortedSet<Tmorton, Tvec, dim>>(morton_codes_set),
            res.morton_leaf_count,
            std::move(res.buf_reduc_index_map),
            std::move(buf_tree_morton));
    }

    template<class Tmorton, class Tvec, u32 dim>
    MortonReducedSet<Tmorton, Tvec, dim> reduce_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSortedSet<Tmorton, Tvec, dim> &&morton_codes_set,
        u32 reduction_level) {

        return reduce_morton_set(
            dev_sched,
            std::forward<MortonCodeSortedSet<Tmorton, Tvec, dim>>(morton_codes_set),
            reduction_level,
            sham::DeviceBuffer<u32>(0, dev_sched),
            sham::DeviceBuffer<Tmorton>(0, dev_sched));
    }

} // namespace shamtree

template class shamtree::MortonCodeSortedSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSortedSet<u64, f64_3, 3>;

template shamtree::MortonReducedSet<u32, f64_3, 3> shamtree::reduce_morton_set<u32, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSortedSet<u32, f64_3, 3> &&morton_codes_set,
    u32 reduction_level);
template shamtree::MortonReducedSet<u64, f64_3, 3> shamtree::reduce_morton_set<u64, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSortedSet<u64, f64_3, 3> &&morton_codes_set,
    u32 reduction_level);

template shamtree::MortonReducedSet<u32, f64_3, 3> shamtree::reduce_morton_set<u32, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSortedSet<u32, f64_3, 3> &&morton_codes_set,
    u32 reduction_level,
    sham::DeviceBuffer<u32> &&cache_buf_reduc_index_map,
    sham::DeviceBuffer<u32> &&cache_reduced_morton_codes);
template shamtree::MortonReducedSet<u64, f64_3, 3> shamtree::reduce_morton_set<u64, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSortedSet<u64, f64_3, 3> &&morton_codes_set,
    u32 reduction_level,
    sham::DeviceBuffer<u32> &&cache_buf_reduc_index_map,
    sham::DeviceBuffer<u64> &&cache_reduced_morton_codes);
