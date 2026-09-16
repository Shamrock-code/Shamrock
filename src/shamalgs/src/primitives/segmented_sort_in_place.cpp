// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file segmented_sort_in_place.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamalgs/primitives/segmented_sort_in_place.hpp"
#include "shambase/alg_primitives.hpp"
#include "shambase/assert.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives::details {

    template<class T, class Comp>
    inline void segmented_sort_in_place_local_insertion_sort(
        sham::DeviceBuffer<T> &buf, const sham::DeviceBuffer<u32> &offsets, Comp &&comp) {

        auto &q = buf.get_dev_scheduler().get_queue();

        size_t interact_count = buf.get_size();
        size_t offsets_count  = offsets.get_size();
        size_t N              = offsets_count - 1;

        sham::kernel_call(
            q,
            sham::MultiRef{offsets},
            sham::MultiRef{buf},
            N,
            [interact_count,
             comp](u32 gid, const u32 *__restrict__ offsets, T *__restrict__ in_out_sorted) {
                u32 start_index = offsets[gid];
                u32 end_index   = offsets[gid + 1];

                // can be equal if there is no interaction for this sender
                SHAM_ASSERT(start_index <= end_index);

                // skip empty ranges to avoid unnecessary work
                if (start_index == end_index) {
                    return;
                }

                // if there is no interactions at the end of the offset list
                // offsets[gid] can be equal to interact_count
                // but we check that start_index != end_index, so here the correct assertions
                // is indeed start_index < interact_count
                SHAM_ASSERT(start_index < interact_count);
                SHAM_ASSERT(end_index <= interact_count); // see the for loop for this one

                shambase::ptr_insert_sort(in_out_sorted, start_index, end_index, comp);
            });
    }

    template<class T, class Comp>
    inline void segmented_sort_in_place_multi_std_sort(
        sham::DeviceBuffer<T> &buf, const sham::DeviceBuffer<u32> &offsets, Comp &&comp) {

        auto &q = buf.get_dev_scheduler().get_queue();

        size_t interact_count = buf.get_size();
        size_t offsets_count  = offsets.get_size();
        size_t N              = offsets_count - 1;

        std::vector<T> buf_stdvec       = buf.copy_to_stdvec();
        std::vector<u32> offsets_stdvec = offsets.copy_to_stdvec();

#pragma omp parallel for
        for (u32 i = 0; i < N; ++i) {
            u32 start_index = offsets_stdvec[i];
            u32 end_index   = offsets_stdvec[i + 1];

            // can be equal if there is no interaction for this sender
            SHAM_ASSERT(start_index <= end_index);

            // skip empty ranges to avoid unnecessary work
            if (start_index == end_index) {
                continue;
            }

            // if there is no interactions at the end of the offset list
            // offsets[gid] can be equal to interact_count
            // but we check that start_index != end_index, so here the correct assertions
            // is indeed start_index < interact_count
            SHAM_ASSERT(start_index < interact_count);
            SHAM_ASSERT(end_index <= interact_count); // see the for loop for this one

            std::sort(buf_stdvec.begin() + start_index, buf_stdvec.begin() + end_index, comp);
        }

        buf.copy_from_stdvec(buf_stdvec);
    }

} // namespace shamalgs::primitives::details

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        /// Sort each segment locally with an insertion sort, one kernel work-item per segment
        struct LocalInsertionSort {
            static constexpr std::string_view variant_type_name = "local_insertion_sort";
        };

        /// Copy back to host and sort each segment with std::sort, parallelized over OpenMP
        struct MultiStdSort {
            static constexpr std::string_view variant_type_name = "multi_std_sort";
        };

        shamalgs::ImplVariantGlobal<LocalInsertionSort, MultiStdSort> segmented_sort_in_place_impl;

        /// Get list of available segmented sort in place implementations
        std::vector<std::string> get_default_impl_list_segmented_sort_in_place() {
            return segmented_sort_in_place_impl.get_default_config_list();
        }

        /// Get the current implementation for segmented sort in place
        std::string get_current_impl_segmented_sort_in_place() {
            return segmented_sort_in_place_impl.get_current_config();
        }

        /// Check if an implementation has been selected for segmented sort in place
        bool is_impl_set_segmented_sort_in_place() { return segmented_sort_in_place_impl.is_set(); }

        /// Set the implementation for segmented sort in place
        void set_impl_segmented_sort_in_place(const std::string &impl) {
            shamlog_info_ln(
                "algs", "setting segmented sort in place implementation to impl :", impl);
            segmented_sort_in_place_impl.set(impl);
        }

        /// Select the default implementation for segmented sort in place
        void autoselect_impl_segmented_sort_in_place() {
            segmented_sort_in_place_impl.set(MultiStdSort{});
            shamlog_info_ln(
                "algs",
                "defaulting segmented sort in place implementation to impl :",
                get_current_impl_segmented_sort_in_place());
        }

    } // namespace impl

    template<class T, class Comp>
    void internal_segmented_sort_in_place(
        sham::DeviceBuffer<T> &buf, const sham::DeviceBuffer<u32> &offsets, Comp &&comp) {

        if (buf.get_size() == 0) {
            return;
        }

        if (offsets.get_size() == 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>("offsets buffer is empty");
        }

        if (!impl::segmented_sort_in_place_impl.is_set()) {
            impl::autoselect_impl_segmented_sort_in_place();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::LocalInsertionSort) {
                    details::segmented_sort_in_place_local_insertion_sort(buf, offsets, comp);
                },
                [&](impl::MultiStdSort) {
                    details::segmented_sort_in_place_multi_std_sort(buf, offsets, comp);
                },
            },
            impl::segmented_sort_in_place_impl.get());
    }

    template<>
    void segmented_sort_in_place<u32_2>(
        sham::DeviceBuffer<u32_2> &buf, const sham::DeviceBuffer<u32> &offsets) {

        internal_segmented_sort_in_place(buf, offsets, [](u32_2 a, u32_2 b) {
            return (a.x() == b.x()) ? (a.y() < b.y()) : (a.x() < b.x());
        });
    }

    template<>
    void segmented_sort_in_place<u32>(
        sham::DeviceBuffer<u32> &buf, const sham::DeviceBuffer<u32> &offsets) {
        internal_segmented_sort_in_place(buf, offsets, [](u32 a, u32 b) {
            return a < b;
        });
    }

} // namespace shamalgs::primitives
