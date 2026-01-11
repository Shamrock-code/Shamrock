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
 */

#include "shambackends/vec.hpp"
#include "shammodels/gsph/solvergraph/edges/GradientsEdge.hpp"
#include "shammodels/gsph/solvergraph/edges/RiemannResultEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    struct RiemannSolverConfig {
        u32 max_iterations      = 100;
        f64 tolerance           = 1e-6;
        bool use_acoustic_guess = true;
    };

    /**
     * @brief Solve Riemann problems at particle interfaces
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     * @tparam RiemannSolverType Riemann solver implementation
     */
    template<class Tvec, template<class> class SPHKernel, class RiemannSolverType>
    class RiemannSolverNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        RiemannSolverType riemann_solver;
        RiemannSolverConfig config;
        bool use_muscl;

        public:
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::FieldRefs<Tvec> &positions_with_ghosts;
            const shamrock::solvergraph::FieldRefs<Tscal> &hpart_with_ghosts;
            const shamrock::solvergraph::Field<Tscal> &density;
            const shamrock::solvergraph::Field<Tscal> &pressure;
            const shamrock::solvergraph::FieldRefs<Tvec> &velocity_with_ghosts;
            const shamrock::solvergraph::Field<Tscal> &soundspeed;
            const GradientsEdge<Tvec> *gradients;
            RiemannResultEdge<Tvec> &riemann_result;
        };

        RiemannSolverNode(
            RiemannSolverType riemann_solver,
            RiemannSolverConfig config = {},
            bool use_muscl             = true)
            : riemann_solver(std::move(riemann_solver)), config(config), use_muscl(use_muscl) {}

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

        std::string _impl_get_label() const override { return "RiemannSolver"; }

        std::string _impl_get_tex() const override {
            return "p^*, v^* \\leftarrow \\text{Riemann}";
        }

        protected:
        void _impl_evaluate_internal() override;
    };

    /// HLL Riemann solver (STUB)
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
            (void) rho_L;
            (void) rho_R;
            (void) cs_L;
            (void) cs_R;
            return {(P_L + P_R) / Tscal{2}, (v_L + v_R) / Tscal{2}};
        }
    };

    /// HLLC Riemann solver (STUB)
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
            (void) rho_L;
            (void) rho_R;
            (void) cs_L;
            (void) cs_R;
            return {(P_L + P_R) / Tscal{2}, (v_L + v_R) / Tscal{2}};
        }
    };

} // namespace shammodels::gsph::solvergraph
