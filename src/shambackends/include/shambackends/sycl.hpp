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
 * @file sycl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "typeAliasBase.hpp" // IWYU pragma: export
#include "typeAliasFp16.hpp" // IWYU pragma: export
#include "typeAliasVec.hpp"  // IWYU pragma: export
#include <sycl/sycl.hpp>     /// IWYU pragma: export

enum SYCLImplementation { ACPP, DPCPP, UNKNOWN };

#ifdef SYCL_COMP_ACPP
constexpr SYCLImplementation sycl_implementation = ACPP;
#else
    #ifdef SYCL_COMP_INTEL_LLVM
constexpr SYCLImplementation sycl_implementation = DPCPP;
    #else
constexpr SYCLImplementation sycl_implementation = UNKNOWN;
    #endif
#endif
