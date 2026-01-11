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
 * @file RiemannResultEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph edge for Riemann solver results
 *
 * Stores the interface state (p*, v*) from Riemann problem solutions.
 * For Newtonian hydro, this is a scalar pressure and velocity.
 * For MHD/relativistic, this would include full flux.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Result from Riemann solver for a particle pair
     *
     * Newtonian hydro version: just p* and v* (scalar along pair axis).
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct RiemannResult {
        Tscal p_star; ///< Interface pressure
        Tscal v_star; ///< Interface velocity (along pair axis)
    };

    /**
     * @brief Edge holding Riemann solver results
     *
     * This edge stores the results of Riemann problems for all particle pairs.
     * Currently a placeholder - the actual storage depends on the neighbor
     * cache structure and how pair data is organized.
     *
     * For now, Riemann results are computed and consumed within UpdateDerivs
     * without explicit storage. This edge is provided for future refactoring
     * to separate Riemann solving from force accumulation.
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    class RiemannResultEdge : public shamrock::solvergraph::IEdgeNamed {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        using Result = RiemannResult<Tscal>;

        RiemannResultEdge(std::string name, std::string tex_symbol)
            : IEdgeNamed(std::move(name), std::move(tex_symbol)) {}

        void free_alloc() override {
            // Placeholder: storage not yet implemented
        }
    };

    // =========================================================================
    // Stubs for extended Riemann results
    // =========================================================================

    /**
     * @brief MHD Riemann result (STUB)
     *
     * For MHD, the Riemann problem returns additional quantities:
     * - B_star: Interface magnetic field
     * - Full flux state for HLLD solver
     */
    template<class Tvec>
    struct MHDRiemannResult {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal p_star;
        Tscal v_star;
        Tvec B_star; ///< Interface magnetic field
        // Additional flux components for HLLD
    };

    /**
     * @brief SR Riemann result (STUB)
     *
     * For special relativity, includes relativistic corrections.
     */
    template<class Tvec>
    struct SRRiemannResult {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal p_star;
        Tscal v_star;
        Tscal W_star; ///< Interface Lorentz factor
        // Full flux for relativistic HLLC
    };

} // namespace shammodels::gsph::solvergraph
