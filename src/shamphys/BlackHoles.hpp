// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BlackHoles.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamunits/Constants.hpp"

namespace shamphys {

    template<class T>
    inline T schwarzschild_radius(T M, T G, T c) {
        return 2 * G * M / (c * c);
    }

    template<class T, class Tu>
    T schwarzschild_radius(T M, const shamunits::UnitSystem<Tu> usys = {}) {
        return schwarzschild_radius(
            M, shamunits::Constants{usys}.G(), shamunits::Constants{usys}.c());
    }

} // namespace shamphys