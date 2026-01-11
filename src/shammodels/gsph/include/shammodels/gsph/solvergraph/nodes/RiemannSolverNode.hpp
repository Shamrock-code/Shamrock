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
 * @file RiemannSolverNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph node for Riemann problem solving
 *
 * Solves Riemann problems at particle pair interfaces to obtain
 * the interface state (p*, v*) used for flux computation.
 *
 * For Newtonian hydro: iterative or approximate Riemann solver
 * For MHD: HLLD solver
 * For SR: relativistic HLLC
 */

#include "shambackends/vec.hpp"
#include "shammodels/gsph/solvergraph/edges/GradientsEdge.hpp"
#include "shammodels/gsph/solvergraph/edges/RiemannResultEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Riemann solver configuration
     */
    struct RiemannSolverConfig {
        /// Maximum iterations for iterative solver
        u32 max_iterations = 100;

        /// Convergence tolerance
        f64 tolerance = 1e-6;

        /// Use acoustic approximation for initial guess
        bool use_acoustic_guess = true;
    };

    /**
     * @brief Solve Riemann problems at particle interfaces
     *
     * For each particle pair (a, b), this node:
     * 1. Reconstructs left/right states at the interface using MUSCL
     * 2. Solves the 1D Riemann problem along the pair axis
     * 3. Returns the interface state (p*, v*)
     *
     * The Riemann solver type depends on the physics:
     * - Newtonian hydro: iterative exact solver (default) or HLL/HLLC
     * - MHD: HLLD solver
     * - SR: relativistic HLLC
     *
     * Inputs:
     * - positions_with_ghosts: Particle positions
     * - hpart_with_ghosts: Smoothing lengths
     * - density, pressure, velocity: Primitive variables
     * - soundspeed: Local sound speed
     * - gradients: MUSCL gradients (optional, for 2nd order)
     * - part_counts: Particle counts per patch
     *
     * Outputs:
     * - riemann_result: Interface states for all pairs
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     * @tparam RiemannSolverType Riemann solver implementation
     */
    template<class Tvec, template<class> class SPHKernel, class RiemannSolverType>
    class RiemannSolverNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        /// Riemann solver instance
        RiemannSolverType riemann_solver;

        /// Solver configuration
        RiemannSolverConfig config;

        /// Whether to use MUSCL reconstruction
        bool use_muscl;

        public:
        /**
         * @brief Edge container for type-safe access
         */
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::FieldRefs<Tvec> &positions_with_ghosts;
            const shamrock::solvergraph::FieldRefs<Tscal> &hpart_with_ghosts;
            const shamrock::solvergraph::Field<Tscal> &density;
            const shamrock::solvergraph::Field<Tscal> &pressure;
            const shamrock::solvergraph::FieldRefs<Tvec> &velocity_with_ghosts;
            const shamrock::solvergraph::Field<Tscal> &soundspeed;
            const GradientsEdge<Tvec> *gradients; // nullable for 1st order
            RiemannResultEdge<Tvec> &riemann_result;
        };

        /**
         * @brief Construct Riemann solver node
         *
         * @param riemann_solver Riemann solver implementation
         * @param config Solver configuration
         * @param use_muscl Whether to use MUSCL reconstruction
         */
        RiemannSolverNode(
            RiemannSolverType riemann_solver,
            RiemannSolverConfig config = {},
            bool use_muscl             = true)
            : riemann_solver(std::move(riemann_solver)), config(config), use_muscl(use_muscl) {}

        /**
         * @brief Set input/output edges (with MUSCL gradients)
         */
        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> velocity_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed,
            std::shared_ptr<GradientsEdge<Tvec>> gradients,
            std::shared_ptr<RiemannResultEdge<Tvec>> riemann_result) {
            __internal_set_ro_edges(
                {part_counts,
                 positions_with_ghosts,
                 hpart_with_ghosts,
                 density,
                 pressure,
                 velocity_with_ghosts,
                 soundspeed,
                 gradients});
            __internal_set_rw_edges({riemann_result});
        }

        /**
         * @brief Set input/output edges (without MUSCL - 1st order)
         */
        void set_edges_first_order(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> velocity_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed,
            std::shared_ptr<RiemannResultEdge<Tvec>> riemann_result) {
            use_muscl = false;
            __internal_set_ro_edges(
                {part_counts,
                 positions_with_ghosts,
                 hpart_with_ghosts,
                 density,
                 pressure,
                 velocity_with_ghosts,
                 soundspeed});
            __internal_set_rw_edges({riemann_result});
        }

        std::string _impl_get_label() const override { return "RiemannSolver"; }

        std::string _impl_get_tex() const override {
            return "p^*, v^* \\leftarrow \\text{Riemann}";
        }

        protected:
        /**
         * @brief Execute the computation
         *
         * Solves Riemann problems for all particle pairs.
         */
        void _impl_evaluate_internal() override;
    };

    // =========================================================================
    // Approximate Riemann Solvers (STUBS)
    // =========================================================================

    /**
     * @brief HLL Riemann solver (STUB)
     *
     * Simple two-wave approximation. Fast but diffusive.
     */
    template<class Tscal>
    struct HLLSolver {
        static RiemannResult<Tscal> solve(
            Tscal rho_L,
            Tscal P_L,
            Tscal v_L,
            Tscal cs_L,
            Tscal rho_R,
            Tscal P_R,
            Tscal v_R,
            Tscal cs_R) {
            // STUB: HLL solver not yet implemented
            return {(P_L + P_R) / Tscal{2}, (v_L + v_R) / Tscal{2}};
        }
    };

    /**
     * @brief HLLC Riemann solver (STUB)
     *
     * Three-wave approximation with contact discontinuity.
     */
    template<class Tscal>
    struct HLLCSolver {
        static RiemannResult<Tscal> solve(
            Tscal rho_L,
            Tscal P_L,
            Tscal v_L,
            Tscal cs_L,
            Tscal rho_R,
            Tscal P_R,
            Tscal v_R,
            Tscal cs_R) {
            // STUB: HLLC solver not yet implemented
            return {(P_L + P_R) / Tscal{2}, (v_L + v_R) / Tscal{2}};
        }
    };

    /**
     * @brief HLLD Riemann solver for MHD (STUB)
     *
     * Five-wave approximation for ideal MHD.
     */
    template<class Tvec>
    struct HLLDSolver {
        using Tscal = shambase::VecComponent<Tvec>;

        static MHDRiemannResult<Tvec> solve(
            Tscal rho_L,
            Tscal P_L,
            Tscal v_L,
            Tscal cs_L,
            Tvec B_L,
            Tscal rho_R,
            Tscal P_R,
            Tscal v_R,
            Tscal cs_R,
            Tvec B_R) {
            // STUB: HLLD solver not yet implemented
            MHDRiemannResult<Tvec> result;
            result.p_star = (P_L + P_R) / Tscal{2};
            result.v_star = (v_L + v_R) / Tscal{2};
            result.B_star = (B_L + B_R) / Tscal{2};
            return result;
        }
    };

    /**
     * @brief Relativistic HLLC solver for SR (STUB)
     */
    template<class Tvec>
    struct SRHLLCSolver {
        using Tscal = shambase::VecComponent<Tvec>;

        static SRRiemannResult<Tvec> solve(
            Tscal rho_L,
            Tscal P_L,
            Tscal v_L,
            Tscal cs_L,
            Tscal W_L,
            Tscal rho_R,
            Tscal P_R,
            Tscal v_R,
            Tscal cs_R,
            Tscal W_R) {
            // STUB: SR HLLC solver not yet implemented
            SRRiemannResult<Tvec> result;
            result.p_star = (P_L + P_R) / Tscal{2};
            result.v_star = (v_L + v_R) / Tscal{2};
            result.W_star = Tscal{1}; // Non-relativistic limit
            return result;
        }
    };

} // namespace shammodels::gsph::solvergraph
