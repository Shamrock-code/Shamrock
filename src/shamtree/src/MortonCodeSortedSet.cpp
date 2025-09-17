// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonCodeSortedSet.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shamalgs/primitives/gen_buffer_index.hpp"
#include <utility>

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set,
        sham::DeviceBuffer<u32> &&cached_map_morton_id_to_obj_id) {

        shammath::AABB<Tvec> bounding_box                = std::move(morton_codes_set.bounding_box);
        u32 cnt_obj                                      = std::move(morton_codes_set.cnt_obj);
        u32 morton_count                                 = std::move(morton_codes_set.morton_count);
        sham::DeviceBuffer<Tmorton> morton_codes_to_sort = std::move(morton_codes_set.morton_codes);

        auto map_morton_id_to_obj_id
            = std::forward<sham::DeviceBuffer<u32>>(cached_map_morton_id_to_obj_id);

        shamalgs::primitives::fill_buffer_index(map_morton_id_to_obj_id, morton_count);

        shamalgs::algorithm::sort_by_key(
            dev_sched, morton_codes_to_sort, map_morton_id_to_obj_id, morton_count);

        return MortonCodeSortedSet<Tmorton, Tvec, dim>(
            std::move(bounding_box),
            std::move(cnt_obj),
            std::move(morton_count),
            std::move(morton_codes_to_sort),
            std::move(map_morton_id_to_obj_id));
    }

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        const sham::DeviceScheduler_ptr &dev_sched,
        MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set) {

        auto map_morton_id_to_obj_id
            = sham::DeviceBuffer<u32>(morton_codes_set.morton_count, dev_sched);

        return sort_morton_set<Tmorton, Tvec, dim>(
            dev_sched,
            std::forward<MortonCodeSet<Tmorton, Tvec, dim>>(morton_codes_set),
            std::move(map_morton_id_to_obj_id));
    }

} // namespace shamtree

template class shamtree::MortonCodeSortedSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSortedSet<u64, f64_3, 3>;

template shamtree::MortonCodeSortedSet<u32, f64_3, 3> shamtree::sort_morton_set<u32, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSet<u32, f64_3, 3> &&morton_codes_set);
template shamtree::MortonCodeSortedSet<u64, f64_3, 3> shamtree::sort_morton_set<u64, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSet<u64, f64_3, 3> &&morton_codes_set);

template shamtree::MortonCodeSortedSet<u32, f64_3, 3> shamtree::sort_morton_set<u32, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSet<u32, f64_3, 3> &&morton_codes_set,
    sham::DeviceBuffer<u32> &&cached_map_morton_id_to_obj_id);
template shamtree::MortonCodeSortedSet<u64, f64_3, 3> shamtree::sort_morton_set<u64, f64_3, 3>(
    const sham::DeviceScheduler_ptr &dev_sched,
    shamtree::MortonCodeSet<u64, f64_3, 3> &&morton_codes_set,
    sham::DeviceBuffer<u32> &&cached_map_morton_id_to_obj_id);
