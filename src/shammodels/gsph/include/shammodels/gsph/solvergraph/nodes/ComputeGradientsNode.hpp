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
     * Uses SPH gradient estimation with omega-corrected kernel gradient
     * and slope limiters to prevent oscillations.
     *
     * @tparam Tvec Vector type
     * @tparam SPHKernel SPH kernel template
     */
    template<class Tvec, template<class> class SPHKernel>
    class ComputeGradientsNode : public shamrock::solvergraph::INode {
        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        u32 limiter_type;
        Tscal limiter_param;

        public:
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

        ComputeGradientsNode(u32 limiter_type = 1, Tscal limiter_param = Tscal{1.5})
            : limiter_type(limiter_type), limiter_param(limiter_param) {}

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

        std::string _impl_get_label() const override { return "ComputeGradients"; }

        std::string _impl_get_tex() const override {
            return "\\nabla\\rho, \\nabla P, \\nabla v \\leftarrow \\text{SPH}";
        }

        protected:
        void _impl_evaluate_internal() override;
    };

} // namespace shammodels::gsph::solvergraph
