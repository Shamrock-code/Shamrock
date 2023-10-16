// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sycl_builtins.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/sycl.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"
#include "shambase/integer.hpp"

namespace shambase {

    /**
     * @brief Fallback function for sycl::any
     * SYCL std : Returns 1 if the most significant bit in any component of x is set; otherwise returns 0.
     * if it is something else than the most significant bit it is UB
     *
     * @tparam T
     * @tparam n
     * @param v
     * @return i32
     */
    template<
        class T,
        int n,
        std::enable_if_t<std::is_integral_v<T>,
                         int> = 0>
    i32 any(sycl::vec<T, n> v) {
        #ifdef SYCL_COMP_INTEL_LLVM
            return sycl::any(v);
        #else
            return shambase::sum_accumulate(
                (v & sycl::vec<T, n>{most_sig_bit_mask<T>()}) >> sycl::vec<T, n>{4}
            );
        #endif
    }

} // namespace shambase