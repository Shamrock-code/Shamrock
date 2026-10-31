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
 * @file ComputeNextRho.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node computing the gas density at the end of the time step
 *
 * Out of place version of the NodeNextRho of the `new_amr_plus_grav` branch of
 * Léodasce Sewanou.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes_with_ghost)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::Field<Tscal>, rho_next)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the gas density at \f$ t^{n+1} \f$,
     * \f$ \rho^{n+1} = \rho^n + \Delta t \, \partial_t \rho \f$, without modifying \f$ \rho^n \f$.
     *
     * `sizes` holds the block counts per patch, each block holding `block_size` cells, and
     * `sizes_with_ghost` the block counts including the ghost blocks, stored after the patch
     * blocks. `rho_next` is resized to `sizes_with_ghost`: the patch cells get
     * \f$ \rho^{n+1} \f$ and the ghost cells keep \f$ \rho^n \f$ (as the in place update of
     * `new_amr_plus_grav`, \f$ \partial_t \rho \f$ being only known on the patch cells).
     */
    template<class Tvec>
    class ComputeNextRho : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;

        public:
        ComputeNextRho(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeNextRho"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
