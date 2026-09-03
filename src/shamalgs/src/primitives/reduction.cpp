// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambase/overloaded.hpp"
#include "fmt/std.h"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/reduction.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"

namespace shamalgs::primitives::impl {

    /// Fallback USM reduction (portable, no group reduction support required)
    struct Fallback {
        static constexpr std::string_view variant_type_name = "fallback";
    };

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
    /// USM group reduction, tunable work-group size
    struct GroupReduction {
        static constexpr std::string_view variant_type_name = "group_reduction";
        u32 group_size                                      = 128;

        /// Expose the group sizes worth benchmarking as separate default implementations
        static std::vector<GroupReduction> variant_custom_defaults() {
            return {
                GroupReduction{16},
                GroupReduction{128},
                GroupReduction{256},
            };
        }
    };
#endif

} // namespace shamalgs::primitives::impl

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
template<>
struct shamalgs::ImplVariantParams<shamalgs::primitives::impl::GroupReduction> {
    static nlohmann::json to_json(const shamalgs::primitives::impl::GroupReduction &p) {
        return {{"group_size", p.group_size}};
    }
    static shamalgs::primitives::impl::GroupReduction from_json(const nlohmann::json &j) {
        shamalgs::primitives::impl::GroupReduction p{};
        if (j.contains("group_size")) {
            p.group_size = j.at("group_size").get<u32>();
        }
        return p;
    }
};
#endif

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        shamalgs::ImplVariantGlobal<
            Fallback
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            ,
            GroupReduction
#endif
            >
            reduction_impl;

        /// Get list of available reduction implementations, as config json strings
        std::vector<std::string> get_default_impl_list_reduction() {
            return reduction_impl.get_default_config_list();
        }

        /// Get the current implementation for reduction, as a config json string
        std::string get_current_impl_reduction() { return reduction_impl.get_current_config(); }

        /// Check if an implementation has been selected for reduction
        bool is_impl_set_reduction() { return reduction_impl.is_set(); }

        /// Set the implementation for reduction, from a config json string
        void set_impl_reduction(const std::string &impl) {
            shamlog_info_ln("algs", "setting reduction implementation to impl :", impl);
            reduction_impl.set(impl);
        }

        /// Select the default implementation for reduction
        void autoselect_impl_reduction() {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            reduction_impl.set(GroupReduction{});
#else
            reduction_impl.set(Fallback{});
#endif
            shamlog_info_ln(
                "algs",
                "defaulting reduction implementation to impl :",
                get_current_impl_reduction());
        }

    } // namespace impl

    template<class T>
    T sum(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        if (!impl::reduction_impl.is_set()) {
            impl::autoselect_impl_reduction();
        }

        return std::visit(
            shambase::overloaded{
                [&](impl::Fallback) {
                    return sum_usm_fallback(sched, buf1, start_id, end_id);
                },
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
                [&](impl::GroupReduction cfg) {
                    return sum_usm_group(sched, buf1, start_id, end_id, cfg.group_size);
                },
#endif
            },
            impl::reduction_impl.get());
    }

    template<class T>
    T min(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        if (!impl::reduction_impl.is_set()) {
            impl::autoselect_impl_reduction();
        }

        return std::visit(
            shambase::overloaded{
                [&](impl::Fallback) {
                    return min_usm_fallback(sched, buf1, start_id, end_id);
                },
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
                [&](impl::GroupReduction cfg) {
                    return min_usm_group(sched, buf1, start_id, end_id, cfg.group_size);
                },
#endif
            },
            impl::reduction_impl.get());
    }

    template<class T>
    T max(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        using namespace shamalgs::reduction::details;

        if (!impl::reduction_impl.is_set()) {
            impl::autoselect_impl_reduction();
        }

        return std::visit(
            shambase::overloaded{
                [&](impl::Fallback) {
                    return max_usm_fallback(sched, buf1, start_id, end_id);
                },
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
                [&](impl::GroupReduction cfg) {
                    return max_usm_group(sched, buf1, start_id, end_id, cfg.group_size);
                },
#endif
            },
            impl::reduction_impl.get());
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
            const sham::DeviceBuffer<_arg_> &buf1,                                                 \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ min<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            const sham::DeviceBuffer<_arg_> &buf1,                                                 \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ max<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            const sham::DeviceBuffer<_arg_> &buf1,                                                 \
            u32 start_id,                                                                          \
            u32 end_id);

    XMAC_TYPES
    #undef X
#endif

} // namespace shamalgs::primitives
