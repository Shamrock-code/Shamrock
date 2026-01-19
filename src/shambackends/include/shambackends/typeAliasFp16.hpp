// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file typeAliasFp16.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#ifdef SYCL_COMP_ACPP

    #include <hipSYCL/sycl/libkernel/vec.hpp>
    #include <hipSYCL/sycl/types.hpp>

// copied from hipsycl sycl/sycl.hpp
namespace sycl {
    using namespace hipsycl::sycl;
}
using f16 = sycl::detail::hp_float; // issue with hipsycl not supporting half

#endif

#ifdef SYCL_COMP_INTEL_LLVM

    #if __has_include(<detail/generic_type_lists.hpp>)
        #include <detail/generic_type_lists.hpp>
        #include <sycl/types.hpp>
        #include <cstdint>
    #else
        #include <sycl/sycl.hpp>
    #endif

using f16 = sycl::half;

#endif

// error with hipsycl half not constexpr
// constexpr f16 operator""_f16(long double n){return f16(n);}
