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
 * @file utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Integer powers with a supplied reciprocal, so a negative exponent never divides
 */

namespace shamunits::details {

    /// Constexpr power expansion function with supplied inverse for negative values
    template<int power, class T>
    inline constexpr T pow_constexpr_fast_inv(T a, T a_inv) noexcept {

        if constexpr (power < 0) {
            return pow_constexpr_fast_inv<-power>(a_inv, a);
        } else if constexpr (power == 0) {
            return T{1};
        } else if constexpr (power % 2 == 0) {
            T tmp = pow_constexpr_fast_inv<power / 2>(a, a_inv);
            return tmp * tmp;
        } else if constexpr (power % 2 == 1) {
            T tmp = pow_constexpr_fast_inv<(power - 1) / 2>(a, a_inv);
            return tmp * tmp * a;
        }
    }

    /**
     * @brief Runtime twin of pow_constexpr_fast_inv, for an exponent only known at run time
     *
     * Same square-and-multiply order (most significant bit first), so for any n it returns
     * exactly what pow_constexpr_fast_inv<n> returns. The caller supplies the reciprocal, so a
     * negative exponent uses a_inv and never divides.
     */
    template<class T>
    inline constexpr T ipow(T a, int n, T a_inv) noexcept {
        T base     = (n < 0) ? a_inv : a;
        unsigned k = (n < 0) ? 0u - static_cast<unsigned>(n) : static_cast<unsigned>(n);

        int top_bit = -1;
        for (unsigned t = k; t != 0; t >>= 1) {
            ++top_bit;
        }

        T res = T{1};
        for (int b = top_bit; b >= 0; --b) {
            res = res * res;
            if (((k >> b) & 1u) != 0) {
                res = res * base;
            }
        }
        return res;
    }
} // namespace shamunits::details
