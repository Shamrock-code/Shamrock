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

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shamcomm/logs.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace shamalgs::primitives::details {

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

    /// Sort the zipped key/value pairs on host using Batcher's odd-even merge sort (reference
    /// implementation, any length)
    template<class Tkey, class Tval>
    void batcher_odd_even_host_serial(std::vector<Tkey> &keys, std::vector<Tval> &values) {

        if (keys.size() != values.size()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the keys and the values must have the same length");
        }

        // Batcher's odd-even merge network, kept as the plain four loops on purpose, this is
        // the readable statement of what the device kernel computes.
        //
        //   for p = 1,2,4,... while p<n
        //     for k = p,p/2,...,1
        //       for j = k mod p to n-1-k step 2k
        //         for i = 0 to min(k-1, n-j-k-1)
        //           if floor((i+j)/2p) == floor((i+j+k)/2p):
        //             compare_exchange(a[i+j], a[i+j+k])

        i32 n = static_cast<i32>(keys.size());
        for (i32 p = 1; p < n; p <<= 1) {
            for (i32 k = p; k >= 1; k >>= 1) {
                for (i32 j = k % p; j <= n - 1 - k; j += 2 * k) {
                    i32 imax = std::min(k - 1, n - j - k - 1);
                    for (i32 i = 0; i <= imax; ++i) {
                        i32 idx1 = i + j;
                        i32 idx2 = i + j + k;
                        if ((idx1 / (2 * p)) == (idx2 / (2 * p))) {
                            if (keys[idx2] < keys[idx1]) {
                                std::swap(keys[idx1], keys[idx2]);
                                std::swap(values[idx1], values[idx2]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Copy both buffers to host, sort the zipped key/value pairs with
    /// batcher_odd_even_host_serial, and copy back
    template<class Tkey, class Tval>
    inline void sort_by_keys_batcher_odd_even_host_serial(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        std::vector<Tkey> key_stdvec = buf_key.copy_to_stdvec();
        std::vector<Tval> val_stdvec = buf_values.copy_to_stdvec();

        std::vector<Tkey> key_sub(key_stdvec.begin(), key_stdvec.begin() + len);
        std::vector<Tval> val_sub(val_stdvec.begin(), val_stdvec.begin() + len);

        batcher_odd_even_host_serial(key_sub, val_sub);

        std::copy(key_sub.begin(), key_sub.end(), key_stdvec.begin());
        std::copy(val_sub.begin(), val_sub.end(), val_stdvec.begin());

        buf_key.copy_from_stdvec(key_stdvec);
        buf_values.copy_from_stdvec(val_stdvec);
    }

} // namespace shamalgs::primitives::details

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        /// Copy the buffers to host, std::sort the zipped key/value pairs, and copy back
        struct StdSort {
            static constexpr std::string_view variant_type_name = "std_sort";
        };

        /// Copy the buffers to host, sort with Batcher's odd-even merge sort, and copy back
        struct BatcherOddEvenHostSerial {
            static constexpr std::string_view variant_type_name = "batcher_odd_even_host_serial";
        };

        shamalgs::ImplVariantGlobal<StdSort, BatcherOddEvenHostSerial> sort_by_keys_impl;

        /// Get list of available sort by keys implementations
        std::vector<std::string> get_default_impl_list_sort_by_keys() {
            return decltype(sort_by_keys_impl)::get_default_config_list();
        }

        /// Get the current implementation for sort by keys
        std::string get_current_impl_sort_by_keys() {
            return sort_by_keys_impl.get_current_config();
        }

        /// Check if an implementation has been selected for sort by keys
        bool is_impl_set_sort_by_keys() { return sort_by_keys_impl.is_set(); }

        /// Set the implementation for sort by keys
        void set_impl_sort_by_keys(const std::string &impl) {
            shamlog_info_ln("algs", "setting sort by keys implementation to impl :", impl);
            sort_by_keys_impl.set(impl);
        }

        /// Select the default implementation for sort by keys
        void autoselect_impl_sort_by_keys() {
            sort_by_keys_impl.set(StdSort{});
            shamlog_info_ln(
                "algs",
                "defaulting sort by keys implementation to impl :",
                get_current_impl_sort_by_keys());
        }

    } // namespace impl

    template<class Tkey, class Tval>
    void sort_by_keys(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        if (!impl::sort_by_keys_impl.is_set()) {
            impl::autoselect_impl_sort_by_keys();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::StdSort) {
                    details::sort_by_keys_std_sort(buf_key, buf_values, len);
                },
                [&](impl::BatcherOddEvenHostSerial) {
                    details::sort_by_keys_batcher_odd_even_host_serial(buf_key, buf_values, len);
                },
            },
            impl::sort_by_keys_impl.get());
    }

    template void sort_by_keys(
        sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<u64> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<f64> &buf_key, sham::DeviceBuffer<f64> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<f32> &buf_key, sham::DeviceBuffer<f32> &buf_values, u32 len);

} // namespace shamalgs::primitives
