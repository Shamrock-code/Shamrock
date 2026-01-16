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
 * @file sycl_builtins.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/vectors.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shambase {

    /**
     * @brief Fallback function for sycl::any
     * SYCL std : Returns 1 if the most significant bit in any component of x is set; otherwise
     * returns 0. if it is something else than the most significant bit it is UB
     *
     * @tparam T
     * @tparam n
     * @param v
     * @return i32
     */
    template<class T, int n, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    i32 any(sycl::vec<T, n> v) {
#ifdef SYCL_COMP_INTEL_LLVM
        return sycl::any(v);
#else
        return sham::sum_accumulate(
            (v & sycl::vec<T, n>{most_sig_bit_mask<T>()}) >> sycl::vec<T, n>{4});
#endif
    }

} // namespace shambase
