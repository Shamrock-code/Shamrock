// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file get_sm_id.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include "shambackends/intrinsics.hpp"
#include "shambackends/intrisics/base_select.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get SM function
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace sham {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE

    DEVICE_ATTRIBUTE_ON_ACPP inline u32 get_sm_id() {
        u32 ret;
    #if __has_builtin(__nvvm_read_ptx_sreg_smid)
        ret = __nvvm_read_ptx_sreg_smid();
    #else
        asm("mov.u32 %0, %%smid;" : "=r"(ret));
    #endif
        return ret;
    }

    // from https://github.com/ROCm/ROCm/issues/2059
    // HW_REG_HW_ID is replaced by HW_REG_HW_ID1 & HW_REG_HW_ID2 on >gfx9
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && defined(__GFX9__)
    #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE
    // from https://github.com/ROCm/ROCm/issues/2059
    DEVICE_ATTRIBUTE_ON_ACPP inline u32 get_sm_id() {
        uint cu_id;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
        return cu_id;
    }

#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && defined(__GFX10__)                   \
    || defined(__GFX11__)
    #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE
    DEVICE_ATTRIBUTE_ON_ACPP inline u32 get_sm_id() {
        uint cu_id;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1, 10, 4)" : "=s"(cu_id));
        return cu_id;
    }

#elif defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST && defined(linux)
    // #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE
    //
    // inline u32 get_sm_id() {
    //    u32 ret;
    //    ret = 2;
    //    return ret;
    //}

#else
    /**
     * @brief Return the SM (Streaming Multiprocessor) ID of the calling thread, or equivalent if
     * implemented.
     */
    inline u32 get_sm_id();
#endif

} // namespace sham
