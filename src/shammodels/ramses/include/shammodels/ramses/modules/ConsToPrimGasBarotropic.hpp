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
 * @file ConsToPrimGasBarotropic.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Conservative to primitive conversion of the gas for a barotropic equation of state
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

// The barotropic pressure only depends on rho, hence no rhoe input
#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_P)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Conservative to primitive conversion of the gas using
     *        shammath::FluidStateBarotropic
     */
    template<class Tvec>
    class NodeConsToPrimGasBarotropic : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        Tscal rho_crit;
        Tscal cs0;
        Tscal gamma;

        public:
        NodeConsToPrimGasBarotropic(u32 block_size, Tscal rho_crit, Tscal cs0, Tscal gamma)
            : block_size(block_size), rho_crit(rho_crit), cs0(cs0), gamma(gamma) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ConsToPrimGasBarotropic"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
