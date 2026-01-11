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
 * @file ComputeOmegaNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph node for computing grad-h correction factor (omega)
 *
 * This node computes the grad-h correction factor omega and density
 * via SPH summation with smoothing length iteration.
 *
 * The computation is physics-independent (same for Newtonian, SR, GR).
 */

#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Compute grad-h correction factor omega and density
     *
     * This node wraps the smoothing length iteration and omega computation.
     * It is physics-independent as it only depends on particle positions
     * and smoothing lengths.
     *
     * Inputs:
     * - positions_with_ghosts: Particle positions (merged with ghosts)
     * - hpart_with_ghosts: Smoothing lengths (merged with ghosts)
     * - neigh_cache: Pre-computed neighbor cache
     * - part_counts: Particle counts per patch
     *
     * Outputs:
     * - omega: Grad-h correction factor
     * - density: SPH-summed density
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class ComputeOmegaNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        /// Particle mass (uniform for now)
        Tscal particle_mass;

        /// Smoothing length tolerance for iteration
        Tscal htol_up_coarse;
        Tscal htol_up_fine;

        /// Convergence parameters
        Tscal epsilon_h;
        u32 h_iter_per_subcycles;

        public:
        /**
         * @brief Edge container for type-safe access
         */
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::FieldRefs<Tvec> &positions_with_ghosts;
            const shamrock::solvergraph::FieldRefs<Tscal> &hpart_with_ghosts;
            shamrock::solvergraph::Field<Tscal> &omega;
            shamrock::solvergraph::Field<Tscal> &density;
        };

        /**
         * @brief Construct compute omega node
         *
         * @param particle_mass Uniform particle mass
         * @param htol_up_coarse Coarse tolerance for h iteration
         * @param htol_up_fine Fine tolerance for h iteration
         * @param epsilon_h Convergence threshold for h
         * @param h_iter_per_subcycles Iterations per subcycle
         */
        ComputeOmegaNode(
            Tscal particle_mass,
            Tscal htol_up_coarse     = Tscal{1.1},
            Tscal htol_up_fine       = Tscal{1.05},
            Tscal epsilon_h          = Tscal{1e-4},
            u32 h_iter_per_subcycles = 30)
            : particle_mass(particle_mass), htol_up_coarse(htol_up_coarse),
              htol_up_fine(htol_up_fine), epsilon_h(epsilon_h),
              h_iter_per_subcycles(h_iter_per_subcycles) {}

        /**
         * @brief Set input/output edges
         */
        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density) {
            __internal_set_ro_edges({part_counts, positions_with_ghosts, hpart_with_ghosts});
            __internal_set_rw_edges({omega, density});
        }

        /**
         * @brief Get typed edge references
         */
        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tvec>>(1),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1)};
        }

        std::string _impl_get_label() const override { return "ComputeOmega"; }

        std::string _impl_get_tex() const override { return "\\Omega, \\rho \\leftarrow h"; }

        protected:
        /**
         * @brief Execute the computation
         *
         * This is a placeholder - the actual implementation calls
         * the existing Solver::compute_omega() method.
         */
        void _impl_evaluate_internal() override;
    };

} // namespace shammodels::gsph::solvergraph
