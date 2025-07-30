// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file get_device_clock.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include "shambackends/intrinsics.hpp"
#include "shambackends/intrisics/base_select.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get device internal clock
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    #define SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE

namespace sham {
    // yeah ok what the heck is this
    // I don't know how to call cuda functions from intel/oneapi device code
    // so I'm just going to use the ptx intrinsics ...
    // But assembly is a piece of crap, so i dug some weird intrinsics out clang's
    // not really documented stuff, like try to google this function you will have fun
    DEVICE_ATTRIBUTE_ON_ACPP inline u64 get_device_clock() {
    #if __has_builtin(__nvvm_read_ptx_sreg_globaltimer)
        return __nvvm_read_ptx_sreg_globaltimer();
    #else
        u64 clock;
        asm("mov.u64 %0, %%globaltimer;" : "=l"(clock));
        return clock;
    #endif
    }
} // namespace sham

#elif defined(_IS_ACPP_SMCP_HOST)
    #define SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE

namespace sham {
    inline u64 get_device_clock() {
        return std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
} // namespace sham

#else
namespace sham {
    /**
     * @brief Return the number of clock cycles elapsed since an arbitrary starting point
     *        on the device.
     */
    inline u64 get_device_clock();
} // namespace sham
#endif
