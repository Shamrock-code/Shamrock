// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ApplyDragExpo.hpp
 * @author Anass Serhani (anass.serhani@cnrs.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the gas-dust drag with the matrix exponential integrator
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho_next)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov_next)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe_next)                                \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho_d_next)                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov_d_next)                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)                                     \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho_d)                                    \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov_d)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Apply the gas-dust drag with the matrix exponential integrator.
     *
     * The momenta at t^{n+1} are the momenta without drag multiplied by the exponential of the
     * drag jacobian (a `ndust + 1` square matrix per cell). The `*_next` inputs are the state at
     * t^{n+1} without drag (U^n + dt dU/dt), the outputs are overwritten with the state at t^{n+1}
     * including the drag.
     *
     * The matrices are stored in local memory when the device has enough of it for a work group of
     * at least 8 cells, and in a global memory scratch buffer otherwise.
     */
    template<class Tvec>
    class ApplyDragExpo : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        u32 ndust;
        /// dust collision rates (the inverse of the stopping times), one per dust species
        std::vector<Tscal> alphas;
        bool enable_frictional_heating;

        public:
        ApplyDragExpo(
            u32 block_size, u32 ndust, std::vector<Tscal> alphas, bool enable_frictional_heating)
            : block_size(block_size), ndust(ndust), alphas(std::move(alphas)),
              enable_frictional_heating(enable_frictional_heating) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ApplyDragExpo"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
