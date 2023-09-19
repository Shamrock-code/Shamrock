// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file slopeLimiter.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Math header to compute slope limiters
 * 
 */

#include "shambase/sycl_utils/sycl_utilities.hpp"

namespace shammath {

    /**
     * @brief Van leer slope limiter
     *
     *
     * \f[
     *   \\phi(f,g) = \frac{f g + \vert f g \vert}{f + g + \epsilon}
     * \f]
     *
     * \todo clarify the definition, normally this is done as adimensionalized function but here it is not.
     *
     * @tparam T 
     * @param f 
     * @param g 
     * @return T 
     */
    template<class T>
    inline T van_leer_slope(T f, T g) {
        T tmp = f * g;
        tmp   = tmp + shambase::sycl_utils::g_sycl_abs(tmp);
        T sm = f + g;
        return tmp / (f + g + 1.e-9*sycl::sign(f));
    }

} // namespace shammath