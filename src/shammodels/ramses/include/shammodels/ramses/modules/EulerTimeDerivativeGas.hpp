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
 * @file EulerTimeDerivativeGas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per cell Euler time derivatives of the gas primitive state
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_press)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_rho)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dx_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dy_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dz_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_P)                                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_vel)                                    \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_press)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the Euler time derivatives of the gas primitive state per cell
     *
     * Those derivatives are the predictor term of the MUSCL-Hancock face
     * reconstruction. Evaluating them once per cell here rather than once per
     * face side in the interpolation nodes avoids re-fetching the velocity
     * gradients for every link.
     */
    template<class Tvec>
    class NodeEulerTimeDerivativeGas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal gamma;

        public:
        NodeEulerTimeDerivativeGas(u32 block_size, Tscal gamma)
            : block_size(block_size), gamma(gamma) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "EulerTimeDerivativeGas"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
