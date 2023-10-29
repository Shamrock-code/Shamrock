// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sycl.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "typeAliasBase.hpp"
#include "typeAliasFp16.hpp"
#include "typeAliasVec.hpp"
#include <sycl/sycl.hpp>

enum SYCLImplementation { OPENSYCL, DPCPP, UNKNOWN };

#ifdef SYCL_COMP_ACPP
constexpr SYCLImplementation sycl_implementation = OPENSYCL;
#else
    #ifdef SYCL_COMP_INTEL_LLVM
constexpr SYCLImplementation sycl_implementation = DPCPP;
    #else
constexpr SYCLImplementation sycl_implementation = UNKNOWN;
    #endif
#endif
