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
 * @file collapse.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Functions for gravitational collapse calculations.
 */

#include "shambackends/math.hpp"
#include "shammath/matrix.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    template<class T>
    T free_fall_time(T rho, T G) {
        return sycl::sqrt(3 * shamunits::pi<T> / (32 * G * rho));
    }

    template<class T>
    T free_fall_time(T rho, const shamunits::UnitSystem<T> usys = {}) {
        return free_fall_time(rho, shamunits::Constants{usys}.G());
    }

} // namespace shamphys
