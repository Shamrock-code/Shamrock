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
 * @file base_select.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file implement the GPU core timeline tool from  A. Richermoz, F. Neyret 2024
 */

#include <shambackends/sycl.hpp>

#if defined(__ACPP__)

    #ifndef ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
        #define ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP 0
    #endif

    #ifndef ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
        #define ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST 0
    #endif

    #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        #define _IS_ACPP_SMCP_CUDA
    #elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
        #define _IS_ACPP_SMCP_HIP
        #if __AMDGCN_WAVEFRONT_SIZE == 64
            #define _IS_ACPP_SMCP_HIP_WAVEFRONT64
        #elif __AMDGCN_WAVEFRONT_SIZE == 32
            #define _IS_ACPP_SMCP_HIP_WAVEFRONT32
        #endif
    #elif defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
        #define _IS_ACPP_SMCP_INTEL_SPIRV
    #elif ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
        #define _IS_ACPP_SSCP
    #elif ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST && !defined(DOXYGEN)
        #define _IS_ACPP_SMCP_HOST
    #endif

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

#if defined(__ACPP__)
    #define DEVICE_ATTRIBUTE_ON_ACPP __device__
#else
    #define DEVICE_ATTRIBUTE_ON_ACPP
#endif
