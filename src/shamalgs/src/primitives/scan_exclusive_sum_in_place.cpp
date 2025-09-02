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
#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"

namespace shamalgs::primitives {

    enum class SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL : u32 { STD_SCAN, DECOUPLED_LOOKBACK_512 };

    SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL get_default_scan_exclusive_sum_in_place_impl() {
#ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        return SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::STD_SCAN;
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512;
    #else
        return SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::STD_SCAN;
    #endif
#endif
    }

    SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL scan_exclusive_sum_in_place_impl
        = get_default_scan_exclusive_sum_in_place_impl();

    void impl::set_impl_scan_exclusive_sum_in_place_default() {
        scan_exclusive_sum_in_place_impl = get_default_scan_exclusive_sum_in_place_impl();
    }

    std::unordered_map<std::string, SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL>
        scan_exclusive_sum_in_place_impl_map
        = {{"std_scan", SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::STD_SCAN},
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
           {"decoupled_lookback_512", SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512}
#endif
    };

    std::vector<std::string> impl::get_impl_list_scan_exclusive_sum_in_place() {
        return shambase::keys_from_map(scan_exclusive_sum_in_place_impl_map);
    }

    template<class T>
    void scan_exclusive_sum_in_place_fallback(sham::DeviceBuffer<T> &buf1, u32 len) {
        auto acc_src = buf1.copy_to_stdvec_idx_range(0, len);
        std::exclusive_scan(acc_src.begin(), acc_src.end(), acc_src.begin(), 0);
        buf1.copy_from_stdvec(acc_src, len);
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
        case SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::STD_SCAN:
            scan_exclusive_sum_in_place_fallback<T>(buf1, len);
            break;
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        case SCAN_EXCLUSIVE_SUM_IN_PLACE_IMPL::DECOUPLED_LOOKBACK_512:
            numeric::details::exclusive_sum_atomic_decoupled_v5_usm_in_place<T, 512>(buf1, len);
            break;
#endif
        default:
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("unimplemented case : {}", u32(scan_exclusive_sum_in_place_impl)));
        }
    }

    template void scan_exclusive_sum_in_place<u32>(sham::DeviceBuffer<u32> &buf1, u32 len);

} // namespace shamalgs::primitives
