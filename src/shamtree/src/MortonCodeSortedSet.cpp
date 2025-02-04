// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonCodeSortedSet.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include <utility>

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim>::MortonCodeSortedSet(
        sham::DeviceScheduler_ptr dev_sched, MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set)
        : bounding_box(std::move(morton_codes_set.bounding_box)),
          cnt_obj(std::move(morton_codes_set.cnt_obj)),
          morton_count(std::move(morton_codes_set.morton_count)),
          sorted_morton_codes(std::move(morton_codes_set.morton_codes)),
          map_morton_id_to_obj_id(
              shamalgs::algorithm::gen_buffer_index_usm(dev_sched, morton_codes_set.morton_count)) {

        shamalgs::algorithm::sort_by_key(
            dev_sched, sorted_morton_codes, map_morton_id_to_obj_id, morton_count);
    }

} // namespace shamtree

template class shamtree::MortonCodeSortedSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSortedSet<u64, f64_3, 3>;
