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
 * @file FieldNames.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Constants for physics field names in GSPH solver
 *
 * This file defines string constants for all physics field names used in the GSPH solver.
 * Newtonian physics uses standard names for consistency with other SPH methods.
 * SR physics will use frame-specific names (e.g., uint_rest, uint_lab).
 */

namespace shammodels::gsph::fields {

    /**
     * @brief Newtonian physics field names
     *
     * Standard field names for Newtonian (non-relativistic) physics.
     * These match the naming convention used by other SPH methods.
     */
    namespace newtonian {

        /// Position field name
        inline constexpr const char *xyz = "xyz";

        /// Velocity field name
        inline constexpr const char *vxyz = "vxyz";

        /// Acceleration field name
        inline constexpr const char *axyz = "axyz";

        /// Smoothing length field name
        inline constexpr const char *hpart = "hpart";

        /// Internal energy field name
        inline constexpr const char *uint = "uint";

        /// Internal energy time derivative field name
        inline constexpr const char *duint = "duint";

        /// Omega (grad-h correction) field name
        inline constexpr const char *omega = "omega";

        /// Density field name
        inline constexpr const char *density = "density";

    } // namespace newtonian

} // namespace shammodels::gsph::fields
