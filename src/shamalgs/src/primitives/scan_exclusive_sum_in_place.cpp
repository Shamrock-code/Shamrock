// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file scan_exclusive_sum_in_place.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the in-place exclusive scan primitive.
 */

#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambase/StlContainerConversion.hpp"
#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include <numeric>

#if defined(__has_include)
    #if __has_include(<AdaptiveCpp/algorithms/numeric.hpp>)
        #include <AdaptiveCpp/algorithms/numeric.hpp>
        #define ACPP_ALG_AVAILABLE
    #endif
#endif

namespace {

#ifdef __ACPP__
    template<class T>
    void scan_exclusive_sum_in_place_std_scan_single_task_acpp(
        sham::DeviceBuffer<T> &buf1, u32 len) {

        auto &q = buf1.get_dev_scheduler_ptr()->get_queue();

        sycl::queue &q_s = q.q;

        if (q_s.is_host()) {
            sham::EventList deps{};
            T *in_out_ptr = buf1.get_write_access(deps);

            auto e = q.submit(deps, [&](sycl::handler &cgh) {
                cgh.single_task([=]() {
                    std::exclusive_scan(in_out_ptr, in_out_ptr + len, in_out_ptr, T{});
                });
            });

            buf1.complete_event_state(e);
        } else {
            auto acc_src = buf1.copy_to_stdvec_idx_range(0, len);
            std::exclusive_scan(acc_src.begin(), acc_src.end(), acc_src.begin(), T{});
            buf1.copy_from_stdvec(acc_src, len);
        }
    }
#endif

    template<class T>
    void scan_exclusive_sum_in_place_fallback(sham::DeviceBuffer<T> &buf1, u32 len) {
        auto acc_src = buf1.copy_to_stdvec_idx_range(0, len);
        std::exclusive_scan(acc_src.begin(), acc_src.end(), acc_src.begin(), 0);
        buf1.copy_from_stdvec(acc_src, len);
    }

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
    template<class T>
    void scan_exclusive_sum_in_place_decoupled_lookback_512(sham::DeviceBuffer<T> &buf1, u32 len) {
        shamalgs::numeric::details::exclusive_sum_atomic_decoupled_v5_usm_in_place<T, 512>(
            buf1, len);
    }
#endif

#ifdef ACPP_ALG_AVAILABLE
    template<class T>
    void scan_exclusive_sum_in_place_adaptivecpp(sham::DeviceBuffer<T> &buf1, u32 len) {
        auto &q = buf1.get_dev_scheduler_ptr()->get_queue().q;

        acpp::algorithms::util::allocation_cache cache{
            acpp::algorithms::util::allocation_type::device};
        acpp::algorithms::util::allocation_group scratch{&cache, q.get_device()};

        sham::DeviceBuffer<T> temp(len, buf1.get_dev_scheduler_ptr());

        sham::EventList deps{};
        const T *in_out_ptr = buf1.get_read_access(deps);
        T *temp_ptr         = temp.get_write_access(deps);

        sycl::event e = adaptivecpp::algorithms::exclusive_scan(
            q, scratch, in_out_ptr, in_out_ptr + len, temp_ptr, T{}, deps.get_events());
        deps.set_consumed(true);

        buf1.complete_event_state(e);
        temp.complete_event_state(e);

        buf1.copy_from(temp, len);
    }
#endif
} // namespace

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        /// std::exclusive_scan on a host copy of the buffer (portable fallback)
        struct StdScan {
            static constexpr std::string_view variant_type_name = "std_scan";
        };

#ifdef __ACPP__
        /// std::exclusive_scan enqueued as a single_task kernel (falls back to a host copy for
        /// non-host queues), AdaptiveCpp-only
        struct StdScanSingleTaskAcpp {
            static constexpr std::string_view variant_type_name = "std_scan_single_task_acpp";
        };
#endif

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        /// Atomic decoupled look-back scan, 512-wide work groups
        struct DecoupledLookback512 {
            static constexpr std::string_view variant_type_name = "decoupled_lookback_512";
        };
#endif

#ifdef ACPP_ALG_AVAILABLE
        /// AdaptiveCpp's own acpp::algorithms::exclusive_scan
        struct AdaptiveCppAlg {
            static constexpr std::string_view variant_type_name = "acpp_alg";
        };
#endif

        shamalgs::ImplVariantGlobal<
            StdScan
#ifdef __ACPP__
            ,
            StdScanSingleTaskAcpp
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            ,
            DecoupledLookback512
#endif
#ifdef ACPP_ALG_AVAILABLE
            ,
            AdaptiveCppAlg
#endif
            >
            scan_exclusive_sum_in_place_impl;

        /// Get list of available scan_exclusive_sum_in_place implementations
        std::vector<std::string> get_default_impl_list_scan_exclusive_sum_in_place() {
            return scan_exclusive_sum_in_place_impl.get_default_config_list();
        }

        /// Get the current implementation for scan_exclusive_sum_in_place
        std::string get_current_impl_scan_exclusive_sum_in_place() {
            return scan_exclusive_sum_in_place_impl.get_current_config();
        }

        /// Check if an implementation has been selected for scan_exclusive_sum_in_place
        bool is_impl_set_scan_exclusive_sum_in_place() {
            return scan_exclusive_sum_in_place_impl.is_set();
        }

        /// Set the implementation for scan_exclusive_sum_in_place
        void set_impl_scan_exclusive_sum_in_place(const std::string &impl) {
            shamlog_info_ln(
                "algs", "setting scan_exclusive_sum_in_place implementation to impl :", impl);
            scan_exclusive_sum_in_place_impl.set(impl);
        }

        /// Select the default implementation for scan_exclusive_sum_in_place
        void autoselect_impl_scan_exclusive_sum_in_place() {
#ifdef __MACH__     // decoupled lookback perf on mac os is awful
    #ifdef __ACPP__ // for acpp we gain using enqueue custom operation instead of copying
            scan_exclusive_sum_in_place_impl.set(StdScanSingleTaskAcpp{});
    #else
            scan_exclusive_sum_in_place_impl.set(StdScan{});
    #endif
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            scan_exclusive_sum_in_place_impl.set(DecoupledLookback512{});
    #else
            scan_exclusive_sum_in_place_impl.set(StdScan{});
    #endif
#endif
            shamlog_info_ln(
                "algs",
                "defaulting scan_exclusive_sum_in_place implementation to impl :",
                get_current_impl_scan_exclusive_sum_in_place());
        }

    } // namespace impl

    template<class T>
    void scan_exclusive_sum_in_place(sham::DeviceBuffer<T> &buf1, u32 len) {

        if (len == 0) {
            return;
        }

        if (len > buf1.get_size()) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "The buffer is smaller than the length of the scan\n"
                "len > buf1.get_size(), len = {}, buf1.get_size() = {}",
                len,
                buf1.get_size()));
        }

        if (!impl::scan_exclusive_sum_in_place_impl.is_set()) {
            impl::autoselect_impl_scan_exclusive_sum_in_place();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::StdScan) {
                    scan_exclusive_sum_in_place_fallback(buf1, len);
                },
#ifdef __ACPP__
                [&](impl::StdScanSingleTaskAcpp) {
                    scan_exclusive_sum_in_place_std_scan_single_task_acpp(buf1, len);
                },
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
                [&](impl::DecoupledLookback512) {
                    scan_exclusive_sum_in_place_decoupled_lookback_512(buf1, len);
                },
#endif
#ifdef ACPP_ALG_AVAILABLE
                [&](impl::AdaptiveCppAlg) {
                    scan_exclusive_sum_in_place_adaptivecpp(buf1, len);
                },
#endif
            },
            impl::scan_exclusive_sum_in_place_impl.get());
    }

    template void scan_exclusive_sum_in_place<u32>(sham::DeviceBuffer<u32> &buf1, u32 len);

} // namespace shamalgs::primitives
