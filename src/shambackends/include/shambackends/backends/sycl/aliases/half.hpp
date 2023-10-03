// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "../check_backend.hpp"

#ifdef SYCL_COMP_ACPP
#include <hipSYCL/sycl/libkernel/vec.hpp>
#include <hipSYCL/sycl/types.hpp>

//copied from hipsycl sycl/sycl.hpp
namespace sycl {
    using namespace hipsycl::sycl;
}

using f16 = u16; // issue with hipsycl not supporting half

#endif

#ifdef SYCL_COMP_INTEL_LLVM

#include <detail/generic_type_lists.hpp>
#include <sycl/types.hpp>

using f16 = sycl::half    ;

#endif

constexpr f16 operator""_f16(long double n){return f16(n);}