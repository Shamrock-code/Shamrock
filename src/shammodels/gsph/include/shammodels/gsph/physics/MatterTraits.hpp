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
 * @file MatterTraits.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Matter model traits for GSPH physics composition
 *
 * Defines the structure of hydrodynamics matter model with field types,
 * static flags for compile-time feature detection, and field name mappings.
 */

#include "shambackends/vec.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"

namespace shammodels::gsph::physics {

    /**
     * @brief Pure hydrodynamics matter model
     *
     * Contains density, velocity, pressure, and internal energy.
     * Suitable for shock tubes, stellar structure, and basic fluid dynamics.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    struct HydroMatter {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        /// Primitive variables (physical quantities)
        struct PrimitiveVars {
            Tscal rho;      ///< Density
            Tvec velocity;  ///< 3-velocity v
            Tscal pressure; ///< Pressure P
            Tscal uint;     ///< Specific internal energy u
        };

        /// Feature flags (compile-time)
        static constexpr bool has_magnetic_field        = false;
        static constexpr bool has_divergence_constraint = false;
        static constexpr bool has_radiation_cooling     = false;
        static constexpr u32 n_evolved_scalars          = 1;

        /// Field names from FieldNames.hpp
        struct FieldNames {
            static constexpr const char *velocity = names::newtonian::vxyz;
            static constexpr const char *uint     = names::newtonian::uint;
            static constexpr const char *density  = names::newtonian::density;
            static constexpr const char *pressure = names::newtonian::pressure;
        };
    };

} // namespace shammodels::gsph::physics
