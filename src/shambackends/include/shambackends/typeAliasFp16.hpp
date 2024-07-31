// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyht(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file typeAliasFp16.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#ifdef SYCL_COMP_ACPP

#include <hipSYCL/sycl/libkernel/vec.hpp>
#include <hipSYCL/sycl/types.hpp>

//copied from hipsycl sycl/sycl.hpp
namespace sycl {
    using namespace hipsycl::sycl;
}
using f16 = sycl::detail::hp_float; // issue with hipsycl not supporting half

#endif

#ifdef SYCL_COMP_INTEL_LLVM
#include <cstdint>
#include <detail/generic_type_lists.hpp>
#include <sycl/types.hpp>
using f16 = sycl::half    ;

#endif

//error with hipsycl half not constexpr
//constexpr f16 operator""_f16(long double n){return f16(n);}

