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
 * @file GRUtils.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambackends/math.hpp"
#include "shamunits/Constants.hpp"
#include <experimental/mdspan>
#include <shambackends/sycl.hpp>

namespace shamphys {

    template<
        class Tvec,
        class Tscal,
        class SizeType,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    struct GR_physics {

        inline static constexpr Tscal GR_dot(
            Tvec a,
            Tvec b,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
        }

        inline static constexpr Tscal get_U0(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
        }

        inline static constexpr Tvec get_V(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
        }

        inline static constexpr Tscal get_sqrt_g(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
        }

        inline static constexpr Tscal get_sqrt_gamma(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            return 0;
        }
    };

} // namespace shamphys
