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
 * @file ComputeGradientsNode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solvergraph node for MUSCL gradient computation
 *
 * Computes SPH gradients of primitive variables for second-order
 * MUSCL reconstruction at particle pair interfaces.
 *
 * Gradients computed:
 * - \nabla \rho (density gradient)
 * - \nabla P (pressure gradient)
 * - \nabla v_x, \nabla v_y, \nabla v_z (velocity gradients)
 *
 * For MHD: additional \nabla B_x, \nabla B_y, \nabla B_z
 */

#include "shambackends/vec.hpp"
#include "shammodels/gsph/solvergraph/edges/GradientsEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief Compute MUSCL reconstruction gradients
     *
     * Uses SPH gradient estimation to compute gradients of primitive
     * variables. These gradients are used for linear reconstruction
     * at particle pair interfaces.
     *
     * The gradient formula uses the omega-corrected kernel gradient:
     * \nabla f_a = (1/\Omega_a) \sum_b (m_b/\rho_b) (f_b - f_a) \nabla W_{ab}
     *
     * Slope limiters are applied to prevent oscillations.
     *
     * Inputs:
     * - positions_with_ghosts: Particle positions (merged with ghosts)
     * - hpart_with_ghosts: Smoothing lengths (merged with ghosts)
     * - omega: Grad-h correction factor
     * - density: SPH density
     * - pressure: Thermodynamic pressure
     * - velocity: Particle velocities (FieldRefs for component access)
     * - part_counts: Particle counts per patch
     *
     * Outputs:
     * - gradients: GradientsEdge containing all gradient fields
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class ComputeGradientsNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        /// Slope limiter type (0 = none, 1 = minmod, 2 = van Leer, 3 = MC)
        u32 limiter_type;

        /// Slope limiter parameter (typically 1.0-2.0)
        Tscal limiter_param;

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
            const shamrock::solvergraph::Field<Tscal> &pressure;
            const shamrock::solvergraph::FieldRefs<Tvec> &velocity_with_ghosts;
            GradientsEdge<Tvec> &gradients;
        };

        /**
         * @brief Construct gradient computation node
         *
         * @param limiter_type Slope limiter type (default: 1 = minmod)
         * @param limiter_param Limiter parameter (default: 1.5)
         */
        ComputeGradientsNode(u32 limiter_type = 1, Tscal limiter_param = Tscal{1.5})
            : limiter_type(limiter_type), limiter_param(limiter_param) {}

        /**
         * @brief Set input/output edges
         */
        void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> velocity_with_ghosts,
            std::shared_ptr<GradientsEdge<Tvec>> gradients) {
            __internal_set_ro_edges(
                {part_counts,
                 positions_with_ghosts,
                 hpart_with_ghosts,
                 omega,
                 density,
                 pressure,
                 velocity_with_ghosts});
            __internal_set_rw_edges({gradients});
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
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(5),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tvec>>(6),
                get_rw_edge<GradientsEdge<Tvec>>(0)};
        }

        std::string _impl_get_label() const override { return "ComputeGradients"; }

        std::string _impl_get_tex() const override {
            return "\\nabla\\rho, \\nabla P, \\nabla v \\leftarrow \\text{SPH}";
        }

        protected:
        /**
         * @brief Execute the computation
         *
         * Computes SPH gradients with slope limiting for MUSCL reconstruction.
         */
        void _impl_evaluate_internal() override;
    };

    // =========================================================================
    // MHD Gradient Node (STUB)
    // =========================================================================

    /**
     * @brief Compute MUSCL gradients for MHD (STUB)
     *
     * Extends hydro gradients with magnetic field gradients:
     * - \nabla B_x, \nabla B_y, \nabla B_z
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class ComputeGradientsMHDNode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        ComputeGradientsMHDNode() = default;

        std::string _impl_get_label() const override { return "ComputeGradients_MHD"; }

        std::string _impl_get_tex() const override {
            return "\\nabla\\rho, \\nabla P, \\nabla v, \\nabla B \\leftarrow \\text{SPH}";
        }

        protected:
        void _impl_evaluate_internal() override {
            // STUB: MHD gradient computation not yet implemented
        }
    };

} // namespace shammodels::gsph::solvergraph
