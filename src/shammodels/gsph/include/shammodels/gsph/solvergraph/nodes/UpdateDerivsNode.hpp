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
 * @file UpdateDerivsNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph node for computing GSPH derivatives
 *
 * Computes acceleration and energy rate from Riemann solver results.
 * This is the core GSPH force computation.
 *
 * For Newtonian hydro:
 *   dv_a/dt = -\sum_b m_b [A_a p*_{ab} \nabla W_a + A_b p*_{ab} \nabla W_b]
 *   du_a/dt = \sum_b m_b A_a p*_{ab} (v*_{ab} - v_a) \cdot \nabla W_a
 *
 * where A_a = 1/(\Omega_a \rho_a^2)
 */

#include "shambackends/vec.hpp"
#include "shammodels/gsph/solvergraph/edges/RiemannResultEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Compute GSPH derivatives from Riemann solutions
     *
     * This node accumulates forces and energy rates over all particle pairs
     * using the Riemann solver interface states.
     *
     * The computation follows the Godunov SPH formulation where forces
     * are computed from the Riemann problem solution at particle interfaces.
     *
     * Inputs:
     * - positions_with_ghosts: Particle positions
     * - hpart_with_ghosts: Smoothing lengths
     * - omega: Grad-h correction factor
     * - density: SPH density
     * - velocity: Particle velocities
     * - riemann_result: Interface states from Riemann solver
     * - part_counts: Particle counts per patch
     *
     * Outputs:
     * - acceleration: dv/dt for each particle
     * - energy_rate: du/dt for each particle
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class UpdateDerivsNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        /// Particle mass (uniform for now)
        Tscal particle_mass;

        public:
        /**
         * @brief Edge container for type-safe access
         */
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::FieldRefs<Tvec> &positions_with_ghosts;
            const shamrock::solvergraph::FieldRefs<Tscal> &hpart_with_ghosts;
            const shamrock::solvergraph::Field<Tscal> &omega;
            const shamrock::solvergraph::Field<Tscal> &density;
            const shamrock::solvergraph::FieldRefs<Tvec> &velocity_with_ghosts;
            shamrock::solvergraph::Field<Tvec> &acceleration;
            shamrock::solvergraph::Field<Tscal> &energy_rate;
        };

        /**
         * @brief Construct derivative computation node
         *
         * @param particle_mass Uniform particle mass
         */
        explicit UpdateDerivsNode(Tscal particle_mass) : particle_mass(particle_mass) {}

        /**
         * @brief Set input/output edges
         */
        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> velocity_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tvec>> acceleration,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> energy_rate) {
            __internal_set_ro_edges(
                {part_counts,
                 positions_with_ghosts,
                 hpart_with_ghosts,
                 omega,
                 density,
                 velocity_with_ghosts});
            __internal_set_rw_edges({acceleration, energy_rate});
        }

        /**
         * @brief Get typed edge references
         */
        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tvec>>(1),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tvec>>(5),
                get_rw_edge<shamrock::solvergraph::Field<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1)};
        }

        std::string _impl_get_label() const override { return "UpdateDerivs"; }

        std::string _impl_get_tex() const override {
            return "\\dot{v}, \\dot{u} \\leftarrow p^*, v^*";
        }

        protected:
        /**
         * @brief Execute the computation
         *
         * Accumulates forces and energy rates from Riemann solutions.
         */
        void _impl_evaluate_internal() override;
    };

    /**
     * @brief UpdateDerivs for SR (STUB)
     *
     * Uses relativistic momentum and energy equations:
     *   d(W v)/dt = relativistic force
     *   d(W - 1 + W u)/dt = relativistic energy rate
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class UpdateDerivsSRNode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            shamrock::solvergraph::Field<Tvec> &d_momentum_dt;
            shamrock::solvergraph::Field<Tscal> &d_energy_dt;
        };

        UpdateDerivsSRNode() = default;

        std::string _impl_get_label() const override { return "UpdateDerivs_SR"; }

        std::string _impl_get_tex() const override {
            return "\\dot{S}, \\dot{E} \\leftarrow \\text{SR-GSPH}";
        }

        protected:
        void _impl_evaluate_internal() override {}
    };

} // namespace shammodels::gsph::solvergraph
