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
 * Defines the structure of matter models:
 * - HydroMatter: Pure hydrodynamics (density, velocity, pressure, internal energy)
 * - MHDMatter: Magnetohydrodynamics (adds B-field, divergence cleaning)
 * - RadHydroMatter: Radiation hydrodynamics (adds cooling/heating)
 *
 * Each matter model specifies:
 * - Field types (primitives, conserved variables)
 * - Static flags for compile-time feature detection
 * - Field name mappings
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"

namespace shammodels::gsph::physics {

    // ========================================================================
    // HydroMatter: Pure hydrodynamics
    // ========================================================================

    /**
     * @brief Pure hydrodynamics matter model
     *
     * Contains density, velocity, pressure, and internal energy.
     * This is the simplest matter model, suitable for:
     * - Shock tubes (Sod, Sedov)
     * - Stellar structure (Lane-Emden)
     * - Basic fluid dynamics
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     */
    template<class Tvec>
    struct HydroMatter {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        /// Primitive variables (physical quantities)
        struct PrimitiveVars {
            Tscal rho;      ///< Density \rho
            Tvec velocity;  ///< 3-velocity v (Newtonian) or spatial 4-velocity (relativistic)
            Tscal pressure; ///< Pressure P
            Tscal uint;     ///< Specific internal energy u
        };

        /// Feature flags (compile-time)
        static constexpr bool has_magnetic_field        = false;
        static constexpr bool has_divergence_constraint = false;
        static constexpr bool has_radiation_cooling     = false;
        static constexpr u32 n_evolved_scalars          = 1; ///< Internal energy only

        /// Field names from FieldNames.hpp
        struct FieldNames {
            static constexpr const char *velocity = names::newtonian::vxyz;
            static constexpr const char *uint     = names::newtonian::uint;
            static constexpr const char *density  = names::newtonian::density;
            static constexpr const char *pressure = names::newtonian::pressure;
        };
    };

    // ========================================================================
    // MHDMatter: Magnetohydrodynamics (STUB)
    // ========================================================================

    /**
     * @brief Magnetohydrodynamics matter model (STUB - not yet implemented)
     *
     * Extends HydroMatter with:
     * - Magnetic field B
     * - Divergence cleaning scalar psi (Dedner et al.)
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct MHDMatter {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        struct PrimitiveVars {
            Tscal rho;
            Tvec velocity;
            Tscal pressure;
            Tscal uint;
            Tvec B;    ///< Magnetic field
            Tscal psi; ///< Dedner cleaning scalar
        };

        static constexpr bool has_magnetic_field        = true;
        static constexpr bool has_divergence_constraint = true;
        static constexpr bool has_radiation_cooling     = false;
        static constexpr u32 n_evolved_scalars          = 2; ///< uint + psi

        /// Dedner cleaning parameters
        struct CleaningParams {
            Tscal ch;      ///< Hyperbolic cleaning wave speed
            Tscal cp;      ///< Parabolic damping coefficient
            Tscal sigma_c; ///< ch/cp ratio parameter (typically 0.18)
        };

        /// Compute cleaning wave speed from local properties
        SYCL_EXTERNAL static Tscal compute_ch(Tscal cs, Tscal v_alfven) {
            return sycl::sqrt(cs * cs + v_alfven * v_alfven);
        }

        /// Alfven velocity
        SYCL_EXTERNAL static Tscal alfven_speed(Tvec B, Tscal rho) {
            return sycl::length(B) / sycl::sqrt(rho);
        }
    };

    // ========================================================================
    // RadHydroMatter: Radiation hydrodynamics (STUB)
    // ========================================================================

    /**
     * @brief Radiation hydrodynamics matter model (STUB - not yet implemented)
     *
     * Adds optically thin cooling/heating to hydrodynamics.
     * Suitable for:
     * - ISM thermal instability (Koyama-Inutsuka, Inoue-Inutsuka)
     * - Stellar atmospheres
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct RadHydroMatter {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        struct PrimitiveVars {
            Tscal rho;
            Tvec velocity;
            Tscal pressure;
            Tscal uint;
            // No explicit radiation field in optically thin limit
            // Cooling is handled as source term in energy equation
        };

        static constexpr bool has_magnetic_field        = false;
        static constexpr bool has_divergence_constraint = false;
        static constexpr bool has_radiation_cooling     = true;
        static constexpr bool is_optically_thin         = true;
        static constexpr u32 n_evolved_scalars          = 1;

        /// Cooling function model selection
        enum class CoolingModel {
            KoyamaInutsuka2000, ///< Equilibrium thermal instability
            InoueInutsuka2006,  ///< Non-equilibrium atomic cooling
            Tabulated           ///< User-provided table
        };
    };

} // namespace shammodels::gsph::physics
