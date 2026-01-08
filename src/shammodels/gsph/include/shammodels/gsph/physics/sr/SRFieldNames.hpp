// -------------------------------------------------------//
// SHAMROCK code - Copyright (c) 2021-2025 Timothée David--Cléris
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// -------------------------------------------------------//

/**
 * @file SRFieldNames.hpp
 * @brief Field name constants for Special Relativistic GSPH physics
 *
 * Physics naming convention (Kitajima et al. 2025):
 * - Lab-frame quantities: N (baryon density), v (velocity), xyz (position)
 * - Rest-frame quantities: n (baryon density), P (pressure), u (internal energy), cs (sound speed)
 * - Lorentz factor: γ (gamma_lor)
 * - EOS gamma: Γ (gamma_eos)
 *
 * Key relation: n_restframe = N_labframe / gamma_lor
 */

#pragma once

#include "shammodels/gsph/FieldNames.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief Field names for SR physics with frame annotations
     */
    namespace fields {

        // ═══════════════════════════════════════════════════════════════════════
        // Import common SPH fields
        // ═══════════════════════════════════════════════════════════════════════
        using shammodels::gsph::fields::HPART;
        using shammodels::gsph::fields::OMEGA;
        using shammodels::gsph::fields::PMASS;

        // ═══════════════════════════════════════════════════════════════════════
        // Lab-frame quantities (computational frame)
        // These use the same storage strings as Newtonian, but have different
        // physical meaning (lab-frame in SR vs single frame in Newtonian)
        // ═══════════════════════════════════════════════════════════════════════

        /// Lab-frame position (same string as Newtonian xyz)
        using shammodels::gsph::fields::XYZ;
        /// Lab-frame velocity (same string as Newtonian vxyz)
        using shammodels::gsph::fields::VXYZ;
        /// Lab-frame acceleration (same string as Newtonian axyz)
        using shammodels::gsph::fields::AXYZ;

        /**
         * @brief Lab-frame baryon density N
         * Computed via SPH kernel summation: N = ν × Σ W(r, h)
         * To get rest-frame density: n = N / γ
         *
         * NOTE: This is DIFFERENT from Newtonian "density" - it's the
         * lab-frame baryon number density, not mass density.
         */
        constexpr const char *N_LABFRAME = "N_labframe";

        // ═══════════════════════════════════════════════════════════════════════
        // Rest-frame quantities (comoving frame of fluid element)
        // ═══════════════════════════════════════════════════════════════════════

        /// Rest-frame pressure (same storage string, SR semantics)
        using shammodels::gsph::computed_fields::PRESSURE;
        /// Rest-frame sound speed (same storage string, SR semantics)
        using shammodels::gsph::computed_fields::SOUNDSPEED;
        /// Rest-frame specific internal energy (same storage string, SR semantics)
        using shammodels::gsph::fields::DUINT;
        using shammodels::gsph::fields::UINT;

        // ═══════════════════════════════════════════════════════════════════════
        // Lorentz factor and SR-specific quantities
        // ═══════════════════════════════════════════════════════════════════════

        constexpr const char *LORENTZ_FACTOR = "lorentz_factor"; ///< γ = 1/√(1 - v²/c²)
        constexpr const char *ENTHALPY       = "enthalpy";       ///< H = 1 + u + P/n
        constexpr const char *V_LABFRAME     = "V_labframe";     ///< Lab-frame volume V = ν/N
        constexpr const char *NU_BARYON      = "nu_baryon";      ///< ν (baryon number per particle)

        // ═══════════════════════════════════════════════════════════════════════
        // Conserved variables (integrated in time)
        // ═══════════════════════════════════════════════════════════════════════

        constexpr const char *S_MOMENTUM  = "S_momentum";  ///< S = γHv (conserved momentum)
        constexpr const char *E_ENERGY    = "e_energy";    ///< e = γH - P/N (conserved energy)
        constexpr const char *DS_MOMENTUM = "dS_momentum"; ///< dS/dt
        constexpr const char *DE_ENERGY   = "de_energy";   ///< de/dt

        // ═══════════════════════════════════════════════════════════════════════
        // Gradients
        // ═══════════════════════════════════════════════════════════════════════

        using shammodels::gsph::computed_fields::GRAD_DENSITY;
        using shammodels::gsph::computed_fields::GRAD_PRESSURE;
        using shammodels::gsph::computed_fields::GRAD_VX;
        using shammodels::gsph::computed_fields::GRAD_VY;
        using shammodels::gsph::computed_fields::GRAD_VZ;

    } // namespace fields

} // namespace shammodels::gsph::physics::sr
