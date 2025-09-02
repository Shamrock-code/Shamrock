// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file reduction.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/primitives/reduction.hpp"
#include "shambase/StlContainerConversion.hpp"
#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "fmt/std.h"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/reduction.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"

namespace shamalgs::primitives {

    enum class REDUCTION_IMPL : u32 { FALLBACK, GROUP_REDUCTION };

    REDUCTION_IMPL get_default_reduction_impl() {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return REDUCTION_IMPL::GROUP_REDUCTION;
#else
        return REDUCTION_IMPL::FALLBACK;
#endif
    }

    REDUCTION_IMPL reduction_impl = get_default_reduction_impl();

    void impl::set_impl_reduction_default() { reduction_impl = get_default_reduction_impl(); }

    std::unordered_map<std::string, REDUCTION_IMPL> reduction_impl_map
        = {{"fallback", REDUCTION_IMPL::FALLBACK},
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
           {"group_reduction", REDUCTION_IMPL::GROUP_REDUCTION}
#endif
    };

    std::vector<std::string> impl::get_impl_list_reduction() {
        return shambase::keys_from_map(reduction_impl_map);
    }

    void impl::set_impl_reduction(const std::string &impl, const std::string &param) {
        shamlog_info_ln("Algs", "Setting reduction implementation to :", impl);
        try {
            reduction_impl = reduction_impl_map.at(impl);
        } catch (const std::out_of_range &e) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "invalid implementation : {}, possible implementations : {}",
                impl,
                get_impl_list_reduction()));
        }
    }

    template<class T>
    T sum(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        switch (reduction_impl) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        case REDUCTION_IMPL::GROUP_REDUCTION:
            return sum_usm_group(sched, buf1, start_id, end_id, 128);
#endif
        case REDUCTION_IMPL::FALLBACK: return sum_usm_fallback(sched, buf1, start_id, end_id);
        default:
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("unimplemented case : {}", u32(reduction_impl)));
        }
    }

    template<class T>
    T min(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        switch (reduction_impl) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        case REDUCTION_IMPL::GROUP_REDUCTION:
            return min_usm_group(sched, buf1, start_id, end_id, 128);
#endif
        case REDUCTION_IMPL::FALLBACK: return min_usm_fallback(sched, buf1, start_id, end_id);
        default:
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("unimplemented case : {}", u32(reduction_impl)));
        }
    }

    template<class T>
    T max(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        switch (reduction_impl) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        case REDUCTION_IMPL::GROUP_REDUCTION:
            return max_usm_group(sched, buf1, start_id, end_id, 128);
#endif
        case REDUCTION_IMPL::FALLBACK: return max_usm_fallback(sched, buf1, start_id, end_id);
        default:
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("unimplemented case : {}", u32(reduction_impl)));
        }
    }

#ifndef DOXYGEN
    #define XMAC_TYPES                                                                             \
        X(f32)                                                                                     \
        X(f32_2)                                                                                   \
        X(f32_3)                                                                                   \
        X(f32_4)                                                                                   \
        X(f32_8)                                                                                   \
        X(f32_16)                                                                                  \
        X(f64)                                                                                     \
        X(f64_2)                                                                                   \
        X(f64_3)                                                                                   \
        X(f64_4)                                                                                   \
        X(f64_8)                                                                                   \
        X(f64_16)                                                                                  \
        X(u32)                                                                                     \
        X(u64)                                                                                     \
        X(i32)                                                                                     \
        X(i64)                                                                                     \
        X(u32_3)                                                                                   \
        X(u64_3)                                                                                   \
        X(i64_3)                                                                                   \
        X(i32_3)

    #define X(_arg_)                                                                               \
        template _arg_ sum<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ min<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ max<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);

    XMAC_TYPES
    #undef X
#endif

} // namespace shamalgs::primitives
