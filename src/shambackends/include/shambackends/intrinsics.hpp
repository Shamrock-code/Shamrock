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
 * @file intrinsics.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include <shambackends/sycl.hpp>

#if defined(__ACPP__) && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    #define _IS_ACPP_SMCP_CUDA
#elif defined(__ACPP__) && defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    #define _IS_ACPP_SMCP_HIP
    #if __AMDGCN_WAVEFRONT_SIZE == 64
        #define _IS_ACPP_SMCP_CUDA_WAVEFRONT64
    #elif __AMDGCN_WAVEFRONT_SIZE == 32
        #define _IS_ACPP_SMCP_CUDA_WAVEFRONT32
    #endif
#elif defined(__ACPP__) && defined(__SYCL_DEVICE_ONLY__)                                           \
    && (defined(__SPIR__) || defined(__SPIRV__))
    #define _IS_ACPP_SMCP_INTEL_SPIRV
#elif defined(__ACPP__) && defined(ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST) && !defined(DOXYGEN)
    #define _IS_ACPP_SMCP_HOST
#endif

#if defined(SYCL_IMPLEMENTATION_ONEAPI) && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    #define _IS_ONEAPI_SMCP_CUDA
#elif defined(SYCL_IMPLEMENTATION_ONEAPI) && defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    #define _IS_ONEAPI_SMCP_HIP
    #if __AMDGCN_WAVEFRONT_SIZE == 64
        #define _IS_ONEAPI_SMCP_HIP_WAVEFRONT64
    #elif __AMDGCN_WAVEFRONT_SIZE == 32
        #define _IS_ONEAPI_SMCP_HIP_WAVEFRONT32
    #endif
#elif defined(SYCL_IMPLEMENTATION_ONEAPI) && defined(__SYCL_DEVICE_ONLY__)                         \
    && (defined(__SPIR__) || defined(__SPIRV__))
    #define _IS_ONEAPI_SMCP_INTEL_SPIRV
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get SM function
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace sham {

#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
    #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE

    __device__ inline u32 get_sm_id() {
        u32 ret;
    #if __has_builtin(__nvvm_read_ptx_sreg_smid)
        ret = __nvvm_read_ptx_sreg_smid();
    #else
        asm("mov.u32 %0, %%smid;" : "=r"(ret));
    #endif
        return ret;
    }

#elif defined(_IS_ONEAPI_SMCP_CUDA)
    #define SHAMROCK_INTRISICS_GET_SMID_AVAILABLE

    inline u32 get_sm_id() {
        u32 ret;
        asm("mov.u32 %0, %%smid;" : "=r"(ret));
        return ret;
    }

#else
    /**
     * @brief Return the SM (Streaming Multiprocessor) ID of the calling thread, or equivalent if
     * implemented.
     */
    inline u32 get_sm_id();
#endif

} // namespace sham

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get device internal clock
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__ACPP__) && ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
    #define SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE

namespace sham {
    __device__ inline u64 get_device_clock() {
    #if __has_builtin(__nvvm_read_ptx_sreg_globaltimer)
        return __nvvm_read_ptx_sreg_globaltimer();
    #else
        u64 clock;
        asm("mov.u64 %0, %%globaltimer;" : "=l"(clock));
        return clock;
    #endif
    }
} // namespace sham
#elif defined(_IS_ONEAPI_SMCP_CUDA)
    #define SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE
namespace sham {
    // yeah ok what the heck is this
    // I don't know how to call cuda functions from intel/oneapi device code
    // so I'm just going to use the ptx intrinsics ...
    // But assembly is a piece of crap, so i dug some weird intrinsics out clang's
    // not really documented stuff, like try to google this function you will have fun
    inline u64 get_device_clock() {
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
