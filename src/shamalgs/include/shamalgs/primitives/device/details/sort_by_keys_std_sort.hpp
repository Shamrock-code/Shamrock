// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sort_by_keys_std_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief std::sort based sort by keys implementation, shared by the sort by keys primitives
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace shamalgs::primitives::device::details {

    /// Copy both buffers to host, std::sort the zipped key/value pairs, and copy back
    template<class Tkey, class Tval>
    inline void sort_by_keys_std_sort(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        std::vector<Tkey> key_stdvec = buf_key.copy_to_stdvec();
        std::vector<Tval> val_stdvec = buf_values.copy_to_stdvec();

        std::vector<std::pair<Tkey, Tval>> zipped(len);
        for (u32 i = 0; i < len; ++i) {
            zipped[i] = std::make_pair(key_stdvec[i], val_stdvec[i]);
        }

        std::sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });

        for (u32 i = 0; i < len; ++i) {
            key_stdvec[i] = zipped[i].first;
            val_stdvec[i] = zipped[i].second;
        }

        buf_key.copy_from_stdvec(key_stdvec);
        buf_values.copy_from_stdvec(val_stdvec);
    }

} // namespace shamalgs::primitives::device::details
