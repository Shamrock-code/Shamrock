// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


/**
 * @file sort_by_keys.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief 
 */

#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"

namespace shamalgs::primitives {

    template<class T>
    void scan_exclusive_sum_in_place(sham::DeviceBuffer<T> &buf1, u32 len){
        auto sched = buf1.get_dev_scheduler_ptr();
        #ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        buf1 =  details::exclusive_sum_fallback_usm(sched, buf1, len);
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
         numeric::details::exclusive_sum_atomic_decoupled_v5_usm_in_place<T, 512>( buf1, len);
    #else
    buf1 =  details::exclusive_sum_fallback_usm(sched, buf1, len);
    #endif
#endif
    }

    template void scan_exclusive_sum_in_place<u32>(sham::DeviceBuffer<u32> &buf1, u32 len);

}
