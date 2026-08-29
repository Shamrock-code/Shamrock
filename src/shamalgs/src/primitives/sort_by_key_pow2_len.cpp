// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sort_by_key_pow2_len.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"
#include "shamalgs/details/algorithm/bitonicSort_updated_usm.hpp"
#include "shamalgs/primitives/sort_by_key_pow2_len.hpp"
#include "shamcomm/logs.hpp"

namespace shamalgs::primitives {

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "Length must be a power of 2");
        }

        if (len < 5e3) {
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback(q, buf_key, buf_values, len);
        } else {
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<Tkey, Tval, 16>(
                q, buf_key, buf_values, len);
        }
    }

    /// namespace to control implementation behavior
    namespace impl {

        /// Bitonic sort, updated USM kernel (see bitonicSort_updated_usm.hpp)
        struct BitonicSort {
            static constexpr std::string_view variant_type_name = "bitonic_sort";
        };

        shamalgs::ImplVariantGlobal<BitonicSort> sort_by_key_pow2_len_impl;

        /// Get list of available sort by key (pow2 len) implementations
        std::vector<std::string> get_default_impl_list_sort_by_key_pow2_len() {
            return sort_by_key_pow2_len_impl.get_default_config_list();
        }

        /// Get the current implementation for sort by key (pow2 len)
        std::string get_current_impl_sort_by_key_pow2_len() {
            return sort_by_key_pow2_len_impl.get_current_config();
        }

        /// Check if an implementation has been selected for sort by key (pow2 len)
        bool is_impl_set_sort_by_key_pow2_len() { return sort_by_key_pow2_len_impl.is_set(); }

        /// Set the implementation for sort by key (pow2 len)
        void set_impl_sort_by_key_pow2_len(const std::string &impl) {
            shamlog_info_ln(
                "algs", "setting sort by key (pow2 len) implementation to impl :", impl);
            sort_by_key_pow2_len_impl.set(impl);
        }

        /// Select the default implementation for sort by key (pow2 len)
        void autoselect_impl_sort_by_key_pow2_len() {
            sort_by_key_pow2_len_impl.set(BitonicSort{});
            shamlog_info_ln(
                "algs",
                "defaulting sort by key (pow2 len) implementation to impl :",
                get_current_impl_sort_by_key_pow2_len());
        }

    } // namespace impl

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "Length must be a power of 2");
        }

        if (!impl::sort_by_key_pow2_len_impl.is_set()) {
            impl::autoselect_impl_sort_by_key_pow2_len();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::BitonicSort) {
                    shamalgs::algorithm::details::sort_by_key_bitonic_updated_usm<Tkey, Tval, 16>(
                        sched, buf_key, buf_values, len);
                },
            },
            impl::sort_by_key_pow2_len_impl.get());
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
