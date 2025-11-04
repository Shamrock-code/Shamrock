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
 * @file utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
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
} // namespace shamunits::details
