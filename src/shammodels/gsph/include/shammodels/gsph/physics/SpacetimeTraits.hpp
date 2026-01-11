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
 * Defines spacetime geometry:
 * - MinkowskiSpacetime: Flat spacetime (Newtonian limit or SR)
 * - SchwarzschildSpacetime: Static black hole (STUB)
 * - KerrSpacetime: Rotating black hole (STUB)
 * - NumericalSpacetime: Tabulated metric from GR solver (STUB)
 *
 * Each spacetime provides:
 * - Metric components (lapse, shift, spatial metric)
 * - Index raising/lowering operations
 * - Geometric source terms (Christoffel contractions)
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::physics {

    // ========================================================================
    // MinkowskiSpacetime: Flat spacetime
    // ========================================================================

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

        /// Dot product with 3-metric: gamma_ij a^i b^j = delta_ij a^i b^j
        SYCL_EXTERNAL static Tscal dot_3metric(Tvec a, Tvec b) { return sycl::dot(a, b); }

        /// Raise index: v^i = gamma^{ij} v_j = delta^{ij} v_j = v_i
        SYCL_EXTERNAL static Tvec raise_index(Tvec v_lower) { return v_lower; }

        /// Lower index: v_i = gamma_{ij} v^j = delta_{ij} v^j = v^i
        SYCL_EXTERNAL static Tvec lower_index(Tvec v_upper) { return v_upper; }

        /// Geometric source terms (none for flat spacetime)
        template<class Derivs, class State>
        SYCL_EXTERNAL static void add_geometric_source(
            Derivs & /* derivs */, Tvec /* x */, const State & /* W */) {
            // No-op for flat spacetime
        }
    };

    // ========================================================================
    // SchwarzschildSpacetime: Static black hole (STUB)
    // ========================================================================

    /**
     * @brief Schwarzschild spacetime (STUB - not yet implemented)
     *
     * Static, spherically symmetric black hole.
     * Metric in Schwarzschild coordinates:
     *   ds^2 = -(1 - r_s/r) dt^2 + (1 - r_s/r)^{-1} dr^2 + r^2 d\Omega^2
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct SchwarzschildSpacetime {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        static constexpr bool is_curved            = true;
        static constexpr bool is_dynamic           = false;
        static constexpr bool needs_metric_storage = false; // Analytic

        Tscal M;     ///< Black hole mass (geometric units G=c=1)
        Tvec center; ///< Black hole position

        SchwarzschildSpacetime(Tscal mass, Tvec center = Tvec{0, 0, 0}) : M(mass), center(center) {}

        /// Lapse: alpha = sqrt(1 - r_s/r)
        SYCL_EXTERNAL Tscal lapse(Tvec x) const {
            Tscal r  = sycl::length(x - center);
            Tscal rs = Tscal{2} * M;
            return sycl::sqrt(Tscal{1} - rs / r);
        }

        /// Shift: beta^i = 0 in Schwarzschild coordinates
        SYCL_EXTERNAL Tvec shift(Tvec /* x */) const { return Tvec{0, 0, 0}; }

        /// Geometric source terms
        template<class Derivs, class State>
        SYCL_EXTERNAL void add_geometric_source(Derivs &derivs, Tvec x, const State &W) const {
            // Christoffel contraction: -Gamma^i_{mu nu} T^{mu nu}
            // Implementation requires full stress-energy tensor
            (void) derivs;
            (void) x;
            (void) W;
            // STUB: Not implemented
        }
    };

    // ========================================================================
    // KerrSpacetime: Rotating black hole (STUB)
    // ========================================================================

    /**
     * @brief Kerr spacetime (STUB - not yet implemented)
     *
     * Rotating black hole with spin parameter a = J/M.
     * Frame dragging effects near the ergosphere.
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct KerrSpacetime {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        static constexpr bool is_curved            = true;
        static constexpr bool is_dynamic           = false;
        static constexpr bool needs_metric_storage = false;

        Tscal M;        ///< Black hole mass
        Tscal a;        ///< Spin parameter (a = J/M, |a| <= M)
        Tvec center;    ///< Black hole position
        Tvec spin_axis; ///< Spin axis direction (unit vector)

        KerrSpacetime(
            Tscal mass, Tscal spin, Tvec center = Tvec{0, 0, 0}, Tvec axis = Tvec{0, 0, 1})
            : M(mass), a(spin), center(center), spin_axis(axis) {}

        // STUB: Metric functions not yet implemented
        SYCL_EXTERNAL Tscal lapse(Tvec /* x */) const { return Tscal{1}; }
        SYCL_EXTERNAL Tvec shift(Tvec /* x */) const { return Tvec{0, 0, 0}; }
    };

    // ========================================================================
    // NumericalSpacetime: Tabulated metric (STUB)
    // ========================================================================

    /**
     * @brief Numerical/tabulated spacetime (STUB - not yet implemented)
     *
     * Metric data interpolated from an external grid, e.g., from a GR
     * evolution code (Einstein Toolkit, BAM, etc.).
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct NumericalSpacetime {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        static constexpr bool is_curved            = true;
        static constexpr bool is_dynamic           = true; // May evolve in time
        static constexpr bool needs_metric_storage = true;

        /// Interpolation methods
        enum class InterpMethod { NGP, CIC, TSC };
        InterpMethod interp_method = InterpMethod::CIC;

        // STUB: Metric field storage would be added here
        // std::shared_ptr<solvergraph::Field<Tscal>> lapse_field;
        // std::shared_ptr<solvergraph::Field<Tvec>> shift_field;
        // etc.
    };

} // namespace shammodels::gsph::physics
