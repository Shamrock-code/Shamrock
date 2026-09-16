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
#include "shamalgs/details/algorithm/batcherOddEvenSort.hpp"
#include "shamalgs/primitives/device/details/sort_by_keys_std_sort.hpp"
#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shamcomm/logs.hpp"
#include <algorithm>
#include <vector>

namespace shamalgs::primitives::details {

    /// Copy both buffers to host, sort the zipped key/value pairs with
    /// batcher_odd_even_host_serial, and copy back
    template<class Tkey, class Tval>
    inline void sort_by_keys_batcher_odd_even_host_serial(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        std::vector<Tkey> key_stdvec = buf_key.copy_to_stdvec();
        std::vector<Tval> val_stdvec = buf_values.copy_to_stdvec();

        std::vector<Tkey> key_sub(key_stdvec.begin(), key_stdvec.begin() + len);
        std::vector<Tval> val_sub(val_stdvec.begin(), val_stdvec.begin() + len);

        algorithm::details::sort_by_key_batcher_odd_even_host_reference(key_sub, val_sub);

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

        /// Copy the buffers to host, sort with Batcher's odd-even merge sort, and copy back
        struct BatcherOddEven {
            static constexpr std::string_view variant_type_name = "batcher_odd_even";
        };

        shamalgs::ImplVariantGlobal<StdSort, BatcherOddEvenHostSerial, BatcherOddEven>
            sort_by_keys_impl;

        /// Get list of available sort by keys implementations
        std::vector<std::string> get_default_impl_list_sort_by_keys() {
            return sort_by_keys_impl.get_default_config_list();
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
                    device::details::sort_by_keys_std_sort(buf_key, buf_values, len);
                },
                [&](impl::BatcherOddEvenHostSerial) {
                    details::sort_by_keys_batcher_odd_even_host_serial(buf_key, buf_values, len);
                },
                [&](impl::BatcherOddEven) {
                    algorithm::details::sort_by_key_batcher_odd_even(
                        buf_key.get_dev_scheduler_ptr(), buf_key, buf_values, len);
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
