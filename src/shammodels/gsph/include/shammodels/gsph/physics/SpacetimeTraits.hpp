// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SpacetimeTraits.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Spacetime traits for GSPH physics composition
 *
 * Defines flat Minkowski spacetime for Newtonian and SR physics.
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::physics {

    /**
     * @brief Flat Minkowski spacetime
     *
     * Used for Newtonian physics (trivial metric) and Special Relativity.
     * All metric operations are identity/trivial - zero overhead.
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct MinkowskiSpacetime {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        /// Feature flags
        static constexpr bool is_curved            = false;
        static constexpr bool is_dynamic           = false;
        static constexpr bool needs_metric_storage = false;

        /// Lapse function alpha = 1
        SYCL_EXTERNAL static Tscal lapse(Tvec /* x */) { return Tscal{1}; }

        /// Shift vector beta^i = 0
        SYCL_EXTERNAL static Tvec shift(Tvec /* x */) { return Tvec{0, 0, 0}; }

        /// Spatial metric determinant sqrt(gamma) = 1
        SYCL_EXTERNAL static Tscal spatial_metric_det(Tvec /* x */) { return Tscal{1}; }

        /// Dot product with 3-metric
        SYCL_EXTERNAL static Tscal dot_3metric(Tvec a, Tvec b) { return sycl::dot(a, b); }

        /// Raise index (identity for flat spacetime)
        SYCL_EXTERNAL static Tvec raise_index(Tvec v_lower) { return v_lower; }

        /// Lower index (identity for flat spacetime)
        SYCL_EXTERNAL static Tvec lower_index(Tvec v_upper) { return v_upper; }
    };

} // namespace shammodels::gsph::physics
