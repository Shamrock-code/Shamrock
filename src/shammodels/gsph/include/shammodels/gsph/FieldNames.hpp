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
 * @brief Common field name constants shared by all GSPH physics modes
 *
 * Physics-specific field names are defined in:
 * - physics/newtonian/NewtonianFieldNames.hpp
 * - physics/sr/SRFieldNames.hpp
 */

namespace shammodels::gsph {

    /**
     * @brief Core SPH fields shared by all physics modes
     */
    namespace fields {

        // ═══════════════════════════════════════════════════════════════════════
        // SPH kernel quantities (truly physics-agnostic)
        // ═══════════════════════════════════════════════════════════════════════
        constexpr const char *HPART = "hpart"; ///< Smoothing length
        constexpr const char *OMEGA = "omega"; ///< Grad-h correction factor Ω
        constexpr const char *PMASS = "pmass"; ///< Per-particle mass/baryon number

        // ═══════════════════════════════════════════════════════════════════════
        // Kinematics (common strings used by PatchDataLayer)
        // These are the actual strings stored in the data layer.
        // Physics modes give these different semantic meaning:
        // - Newtonian: position, velocity, acceleration (single frame)
        // - SR: lab-frame position, lab-frame velocity, lab-frame acceleration
        // ═══════════════════════════════════════════════════════════════════════
        constexpr const char *XYZ  = "xyz";  ///< Position field name
        constexpr const char *VXYZ = "vxyz"; ///< Velocity field name
        constexpr const char *AXYZ = "axyz"; ///< Acceleration field name

        // ═══════════════════════════════════════════════════════════════════════
        // Energy (common strings, physics gives meaning)
        // ═══════════════════════════════════════════════════════════════════════
        constexpr const char *UINT  = "uint";  ///< Specific internal energy
        constexpr const char *DUINT = "duint"; ///< Time derivative of internal energy

    } // namespace fields

    /**
     * @brief Common computed field names (physics-agnostic)
     */
    namespace computed_fields {

        // ═══════════════════════════════════════════════════════════════════════
        // Thermodynamic outputs (common strings for VTK output)
        // ═══════════════════════════════════════════════════════════════════════
        constexpr const char *DENSITY    = "density";    ///< Output density field name
        constexpr const char *PRESSURE   = "pressure";   ///< Pressure
        constexpr const char *SOUNDSPEED = "soundspeed"; ///< Sound speed

        // ═══════════════════════════════════════════════════════════════════════
        // Gradients for MUSCL reconstruction
        // ═══════════════════════════════════════════════════════════════════════
        constexpr const char *GRAD_DENSITY  = "grad_density";
        constexpr const char *GRAD_PRESSURE = "grad_pressure";
        constexpr const char *GRAD_VX       = "grad_vx";
        constexpr const char *GRAD_VY       = "grad_vy";
        constexpr const char *GRAD_VZ       = "grad_vz";

    } // namespace computed_fields

} // namespace shammodels::gsph
