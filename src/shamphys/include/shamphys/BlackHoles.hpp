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
 * @file BlackHoles.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamunits/UnitSystem.hpp"
#include "shamunits/constant_table.hpp"

namespace shamphys {

    template<class T>
    inline T schwarzschild_radius(T M, T G, T c) {
        return 2 * G * M / (c * c);
    }

    template<class T, class Tu>
    T schwarzschild_radius(T M, const shamunits::UnitSystem<Tu> usys = {}) {
        return schwarzschild_radius(
            M,
            usys.template get<shamunits::constants::G>(),
            usys.template get<shamunits::constants::c>());
    }

} // namespace shamphys
