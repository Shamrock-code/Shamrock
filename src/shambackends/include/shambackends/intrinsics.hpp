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

#if defined(__ACPP__) && defined(ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA) && !defined(DOXYGEN)
    #define _IS_ACPP_SMCP_CUDA
#endif

#if defined(__ACPP__) && defined(ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST) && !defined(DOXYGEN)
    #define _IS_ACPP_SMCP_HOST
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get SM function
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_IS_ACPP_SMCP_CUDA)

namespace sham {

    ACPP_UNIVERSAL_TARGET
    uint get_sm_id() {
        uint32_t ret;
        __acpp_if_target_cuda(asm("mov.u32 %0, %%smid;" : "=r"(ret)));
        return ret;
    }

} // namespace sham

#elif defined(_IS_ACPP_SMCP_HOST)
    #define INTRISICS_GET_SM_DEFINED

namespace sham {

    ACPP_UNIVERSAL_TARGET
    uint get_sm_id() {
        int core_id;
        core_id = sched_getcpu();
        return core_id;
    }

} // namespace sham
#else

namespace sham {

    /**
     * @brief Return the SM (Streaming Multiprocessor) ID of the calling thread if on a NVIDIA GPU.
     * @return The SM ID of the calling thread if on a NVIDIA GPU, 0 otherwise.
     *
     * This is a shamrock wrapper for the CUDA intrinsics __acpp_if_target_cuda(asm("mov.u32 %0,
     * %%smid;" : "=r"(ret)));
     */
    inline uint get_sm_id() { return 0; }

} // namespace sham

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// Get device internal clock
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_IS_ACPP_SMCP_CUDA)

    #include <cuda/std/chrono>

namespace sham {

    ACPP_UNIVERSAL_TARGET inline u64 get_device_clock() {
        u64 out = 0;

        __acpp_if_target_cuda(
            out = cuda::std::chrono::high_resolution_clock::now().time_since_epoch().count(););

        return out;
    }

} // namespace sham
#elif defined(_IS_ACPP_SMCP_HOST)
    #define INTRISICS_GET_CLOCK_DEFINED
namespace sham {

    ACPP_UNIVERSAL_TARGET inline u64 get_device_clock() {

        u64 val = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << val << std::endl;
        return val;
    }

} // namespace sham
#else
namespace sham {

    /**
     * @brief Return the number of clock cycles elapsed since an arbitrary starting point
     *        on the device.
     */
    inline u64 get_device_clock() { return 0; }

} // namespace sham
#endif
