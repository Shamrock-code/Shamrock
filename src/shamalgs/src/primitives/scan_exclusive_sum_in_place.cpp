// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file scan_exclusive_sum_in_place.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the in-place exclusive scan primitive.
 */

#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambase/StlContainerConversion.hpp"
#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"

#if defined(__has_include)
    #if __has_include(<AdaptiveCpp/algorithms/numeric.hpp>)
        #include <AdaptiveCpp/algorithms/numeric.hpp>
        #define ACPP_ALG_AVAILABLE
    #endif
#endif

namespace {

#ifdef __ACPP__
    template<class T>
    void scan_exclusive_sum_in_place_std_scan_custom_op_acpp(sham::DeviceBuffer<T> &buf1, u32 len) {

        auto &q = buf1.get_dev_scheduler_ptr()->get_queue();

        sycl::queue &q_s = q.q;

        if (q_s.is_host()) {
            sham::EventList deps{};
            T *in_out_ptr = buf1.get_write_access(deps);

            auto e = q.submit(deps, [&](sycl::handler &cgh) {
                cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle &h) {
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

    enum class EXSCAN_IN_PLACE_IMPL : u32 {
        STD_SCAN,
#ifdef __ACPP__
        STD_SCAN_CUSTOM_OP_ACPP,
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        DECOUPLED_LOOKBACK_512,
#endif
#ifdef ACPP_ALG_AVAILABLE
        ADAPTIVECPP_ALG,
#endif
    };

    EXSCAN_IN_PLACE_IMPL get_default_scan_exclusive_sum_in_place_impl() {
#ifdef __MACH__     // decoupled lookback perf on mac os is awfull
    #ifdef __ACPP__ // for acpp we gain using enqueue custom operation instead of copying
        return EXSCAN_IN_PLACE_IMPL::STD_SCAN_CUSTOM_OP_ACPP;
    #else
        return EXSCAN_IN_PLACE_IMPL::STD_SCAN;
    #endif
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return EXSCAN_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512;
    #else
        return EXSCAN_IN_PLACE_IMPL::STD_SCAN;
    #endif
#endif
    }

    EXSCAN_IN_PLACE_IMPL scan_exclusive_sum_in_place_impl
        = get_default_scan_exclusive_sum_in_place_impl();

    inline EXSCAN_IN_PLACE_IMPL scan_exclusive_sum_in_place_impl_from_params(
        const std::string &impl) {
        if (impl == "std_scan") {
            return EXSCAN_IN_PLACE_IMPL::STD_SCAN;
#ifdef __ACPP__
        } else if (impl == "std_scan_custom_op_acpp") {
            return EXSCAN_IN_PLACE_IMPL::STD_SCAN_CUSTOM_OP_ACPP;
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        } else if (impl == "decoupled_lookback_512") {
            return EXSCAN_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512;
#endif
#ifdef ACPP_ALG_AVAILABLE
        } else if (impl == "acpp_alg") {
            return EXSCAN_IN_PLACE_IMPL::ADAPTIVECPP_ALG;
#endif
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
            "invalid implementation : {}, possible implementations : {}",
            impl,
            impl::get_default_impl_list_scan_exclusive_sum_in_place()));
    }

    inline shamalgs::impl_param scan_exclusive_sum_in_place_impl_to_params(
        const EXSCAN_IN_PLACE_IMPL &impl) {
        if (impl == EXSCAN_IN_PLACE_IMPL::STD_SCAN) {
            return {"std_scan", ""};
#ifdef __ACPP__
        } else if (impl == EXSCAN_IN_PLACE_IMPL::STD_SCAN_CUSTOM_OP_ACPP) {
            return {"std_scan_custom_op_acpp", ""};
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        } else if (impl == EXSCAN_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512) {
            return {"decoupled_lookback_512", ""};
#endif
#ifdef ACPP_ALG_AVAILABLE
        } else if (impl == EXSCAN_IN_PLACE_IMPL::ADAPTIVECPP_ALG) {
            return {"acpp_alg", ""};
#endif
        }

        throw shambase::make_except_with_loc<std::invalid_argument>(
            shambase::format("unknow scan_exclusive_sum_in_place implementation : {}", u32(impl)));
    }

    std::vector<shamalgs::impl_param> impl::get_default_impl_list_scan_exclusive_sum_in_place() {
        return {
            {"std_scan", ""},
#ifdef __ACPP__
            {"std_scan_custom_op_acpp", ""},
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            {"decoupled_lookback_512", ""},
#endif
#ifdef ACPP_ALG_AVAILABLE
            {"acpp_alg", ""},
#endif
        };
    }

    shamalgs::impl_param impl::get_current_impl_scan_exclusive_sum_in_place() {
        return scan_exclusive_sum_in_place_impl_to_params(scan_exclusive_sum_in_place_impl);
    }

    void impl::set_impl_scan_exclusive_sum_in_place(
        const std::string &impl, const std::string &param) {
        shamlog_info_ln(
            "tree", "setting scan_exclusive_sum_in_place implementation to impl :", impl);
        scan_exclusive_sum_in_place_impl = scan_exclusive_sum_in_place_impl_from_params(impl);
    }

    template<class T>
    void scan_exclusive_sum_in_place(sham::DeviceBuffer<T> &buf1, u32 len) {

        if (len == 0) {
            return;
        }

        if (len > buf1.get_size()) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "The buffer is smaller than the length of the scan\n"
                "len > buf1.get_size(), len = {}, buf1.get_size() = {}",
                len,
                buf1.get_size()));
        }

        switch (scan_exclusive_sum_in_place_impl) {
        case EXSCAN_IN_PLACE_IMPL::STD_SCAN: scan_exclusive_sum_in_place_fallback(buf1, len); break;
#ifdef __ACPP__
        case EXSCAN_IN_PLACE_IMPL::STD_SCAN_CUSTOM_OP_ACPP:
            scan_exclusive_sum_in_place_std_scan_custom_op_acpp(buf1, len);
            break;
#endif
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        case EXSCAN_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512:
            scan_exclusive_sum_in_place_decoupled_lookback_512(buf1, len);
            break;
#endif
#ifdef ACPP_ALG_AVAILABLE
        case EXSCAN_IN_PLACE_IMPL::ADAPTIVECPP_ALG:
            scan_exclusive_sum_in_place_adaptivecpp(buf1, len);
            break;
#endif
        default:
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("unimplemented case : {}", u32(scan_exclusive_sum_in_place_impl)));
        }
    }

    template void scan_exclusive_sum_in_place<u32>(sham::DeviceBuffer<u32> &buf1, u32 len);

} // namespace shamalgs::primitives
