// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file bitonicSort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"

namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval>
    void sort_by_key_bitonic_legacy(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    // implementation disabled since it exibit the same performance as the normal one
    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated_xor_swap(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    template<class Tkey, class Tval>
    inline void sort_by_key_bitonic_fallback(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {
        std::vector<std::pair<Tkey, Tval>> v;
        v.resize(len);

        {
            sycl::host_accessor key{buf_key, sycl::read_only};
            sycl::host_accessor vals{buf_values, sycl::read_only};

            for (u32 i = 0; i < len; i++) {
                v[i] = {key[i], vals[i]};
            }
        }

        std::sort(v.begin(), v.end());

        {
            sycl::host_accessor key{buf_key, sycl::write_only};
            sycl::host_accessor vals{buf_values, sycl::write_only};

            for (u32 i = 0; i < len; i++) {
                key[i]  = v[i].first;
                vals[i] = v[i].second;
            }
        }
    }

} // namespace shamalgs::algorithm::details
