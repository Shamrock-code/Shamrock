// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonReducedSet.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamtree/MortonReducedSet.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include <utility>

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    MortonReducedSet<Tmorton, Tvec, dim> reduce_morton_set(
        sham::DeviceScheduler_ptr dev_sched,
        MortonCodeSortedSet<Tmorton, Tvec, dim> &&morton_codes_set,
        u32 reduction_level) {}

} // namespace shamtree

template class shamtree::MortonCodeSortedSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSortedSet<u64, f64_3, 3>;

template shamtree::MortonReducedSet<u32, f64_3, 3> shamtree::reduce_morton_set<u32, f64_3, 3>(
    sham::DeviceScheduler_ptr dev_sched,
    shamtree::MortonCodeSortedSet<u32, f64_3, 3> &&morton_codes_set,
    u32 reduction_level);
template shamtree::MortonReducedSet<u64, f64_3, 3> shamtree::reduce_morton_set<u64, f64_3, 3>(
    sham::DeviceScheduler_ptr dev_sched,
    shamtree::MortonCodeSortedSet<u64, f64_3, 3> &&morton_codes_set,
    u32 reduction_level);
