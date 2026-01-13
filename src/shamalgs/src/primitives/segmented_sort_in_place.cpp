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

        enum class SEGMENTED_SORT_IN_PLACE_IMPL : u32 {
            LOCAL_INSERTION_SORT,
            MULTI_STD_SORT,
        };

        SEGMENTED_SORT_IN_PLACE_IMPL get_default_segmented_sort_in_place_impl() {
            return SEGMENTED_SORT_IN_PLACE_IMPL::MULTI_STD_SORT;
        }

        SEGMENTED_SORT_IN_PLACE_IMPL segmented_sort_in_place_impl
            = get_default_segmented_sort_in_place_impl();

        inline SEGMENTED_SORT_IN_PLACE_IMPL segmented_sort_in_place_impl_from_params(
            const std::string &impl) {
            if (impl == "local_insertion_sort") {
                return SEGMENTED_SORT_IN_PLACE_IMPL::LOCAL_INSERTION_SORT;
            } else if (impl == "multi_std_sort") {
                return SEGMENTED_SORT_IN_PLACE_IMPL::MULTI_STD_SORT;
            }
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "invalid implementation : {}, possible implementations : {}",
                impl,
                impl::get_default_impl_list_segmented_sort_in_place()));
        }

        inline shamalgs::impl_param segmented_sort_in_place_impl_to_params(
            const SEGMENTED_SORT_IN_PLACE_IMPL &impl) {
            if (impl == SEGMENTED_SORT_IN_PLACE_IMPL::LOCAL_INSERTION_SORT) {
                return {"local_insertion_sort", ""};
            } else if (impl == SEGMENTED_SORT_IN_PLACE_IMPL::MULTI_STD_SORT) {
                return {"multi_std_sort", ""};
            }
            throw shambase::make_except_with_loc<std::invalid_argument>(
                shambase::format("unknow segmented sort in place implementation : {}", u32(impl)));
        }

        /// Get list of available segmented sort in place implementations
        std::vector<shamalgs::impl_param> get_default_impl_list_segmented_sort_in_place() {
            return {
                {"local_insertion_sort", ""},
                {"multi_std_sort", ""},
            };
        }

        /// Get the current implementation for segmented sort in place
        shamalgs::impl_param get_current_impl_segmented_sort_in_place() {
            return segmented_sort_in_place_impl_to_params(segmented_sort_in_place_impl);
        }

        /// Set the implementation for segmented sort in place
        void set_impl_segmented_sort_in_place(const std::string &impl, const std::string &param) {
            shamlog_info_ln(
                "tree", "setting segmented sort in place implementation to impl :", impl);
            segmented_sort_in_place_impl = segmented_sort_in_place_impl_from_params(impl);
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

        switch (impl::segmented_sort_in_place_impl) {
        case impl::SEGMENTED_SORT_IN_PLACE_IMPL::LOCAL_INSERTION_SORT:
            details::segmented_sort_in_place_local_insertion_sort(buf, offsets, comp);
            break;

        case impl::SEGMENTED_SORT_IN_PLACE_IMPL::MULTI_STD_SORT:
            details::segmented_sort_in_place_multi_std_sort(buf, offsets, comp);
            break;
        default:
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "unimplemented case : {}", u32(impl::segmented_sort_in_place_impl)));
        }
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
