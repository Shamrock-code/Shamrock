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
 * @file NewtonianFieldNames.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Field name constants for Newtonian GSPH physics
 *
 * For Newtonian physics, there is no frame distinction - all quantities
 * are in the single inertial frame. No suffixes needed.
 */

#include "shammodels/gsph/FieldNames.hpp"

namespace shammodels::gsph::physics::newtonian {

    /**
     * @brief Field names for Newtonian physics (no frame suffix needed)
     */
    namespace fields {

        // Import common SPH kernel fields
        using shammodels::gsph::fields::HPART;
        using shammodels::gsph::fields::OMEGA;
        using shammodels::gsph::fields::PMASS;

        // Position and kinematics (single frame)
        constexpr const char *XYZ  = "xyz";  ///< Position
        constexpr const char *VXYZ = "vxyz"; ///< Velocity
        constexpr const char *AXYZ = "axyz"; ///< Acceleration

        // Energy (single frame)
        constexpr const char *UINT  = "uint";  ///< Specific internal energy
        constexpr const char *DUINT = "duint"; ///< Time derivative of internal energy

        // Density (SPH kernel summation: ρ = m × Σ W)
        constexpr const char *DENSITY = "density";

        // Thermodynamics
        using shammodels::gsph::computed_fields::PRESSURE;
        using shammodels::gsph::computed_fields::SOUNDSPEED;

        // Gradients
        using shammodels::gsph::computed_fields::GRAD_DENSITY;
        using shammodels::gsph::computed_fields::GRAD_PRESSURE;

    } // namespace fields

} // namespace shammodels::gsph::physics::newtonian
