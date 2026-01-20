// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sort_by_keys.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"
#include "shamalgs/details/algorithm/bitonicSort_updated_usm.hpp"

namespace shamalgs::primitives {

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        if (len < 5e3) {
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback(q, buf_key, buf_values, len);
        } else {
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<Tkey, Tval, 16>(
                q, buf_key, buf_values, len);
        }
    }

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {
        shamalgs::algorithm::details::sort_by_key_bitonic_updated_usm<Tkey, Tval, 16>(
            sched, buf_key, buf_values, len);
    }

    template void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<f64> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<f32> &buf_values,
        u32 len);

} // namespace shamalgs::primitives
