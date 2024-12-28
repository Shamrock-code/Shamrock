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
 * @file group_reduc_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

#ifdef SYCL_COMP_INTEL_LLVM
    #define SYCL_SUM_OP                                                                            \
        sycl::plus<> {}
    #define SYCL_MIN_OP                                                                            \
        sycl::minimum<> {}
    #define SYCL_MAX_OP                                                                            \
        sycl::maximum<> {}
#endif

#ifdef SYCL_COMP_ACPP
template<typename T = void>
struct _tmp_max {
    HIPSYCL_UNIVERSAL_TARGET inline T operator()(const T &lhs, const T &rhs) const {
        return sham::max(lhs, rhs);
    }
};
template<typename T = void>
struct _tmp_min {
    HIPSYCL_UNIVERSAL_TARGET inline T operator()(const T &lhs, const T &rhs) const {
        return sham::min(lhs, rhs);
    }
};
    #define SYCL_SUM_OP                                                                            \
        sycl::plus<T> {}
    #define SYCL_MIN_OP                                                                            \
        _tmp_min<T> {}
    #define SYCL_MAX_OP                                                                            \
        _tmp_max<T> {}
#endif

#ifdef SYCL_COMP_SYCLUNKNOWN
template<typename T = void>
struct _tmp_max {
    inline T operator()(const T &lhs, const T &rhs) const { return sham::max(lhs, rhs); }
};
template<typename T = void>
struct _tmp_min {
    inline T operator()(const T &lhs, const T &rhs) const { return sham::max(lhs, rhs); }
};
    #define SYCL_SUM_OP                                                                            \
        sycl::plus<T> {}
    #define SYCL_MIN_OP                                                                            \
        _tmp_min<T> {}
    #define SYCL_MAX_OP                                                                            \
        _tmp_max<T> {}
#endif
