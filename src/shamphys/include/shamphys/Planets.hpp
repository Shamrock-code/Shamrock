// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Planets.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    template<class T>
    T hill_radius(T R, T m, T M) {
        return R * sycl::cbrt(m / (3 * M));
    }

    template<class T>
    T keplerian_speed(T G, T M, T R) {
        return sycl::sqrt(G * M / R);
    }

    template<class T, class Tu>
    T keplerian_speed(T M, T R, const shamunits::UnitSystem<Tu> usys = {}) {
        return keplerian_speed(shamunits::Constants{usys}.G(), M, R);
    }

} // namespace shamphys