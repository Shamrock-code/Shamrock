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
 * @file ComputeAnalyticalGravity.hpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula.
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    class NodeComputeAnalyticalGravity : public shamrock::solvergraph::INode {

        using AGConfig = typename shammodels::basegodunov::AnalyticalGravityConfig<Tvec>;
        using Tscal    = shambase::VecComponent<Tvec>;

        u32 block_size;
        AGConfig config;

        public:
        NodeComputeAnalyticalGravity(u32 block_size, AGConfig config)
            : block_size(block_size), config(config) {}

#define NODE_COMPUTE_ANALYTICAL_GRAVITY(X_RO, X_RW)                                                \
    /* inputs */                                                                                   \
    X_RO(                                                                                          \
        shamrock::solvergraph::Indexes<u32>,                                                       \
        sizes) /* number of blocks per patch for all patches on the current MPI process*/          \
    X_RO(                                                                                          \
        shamrock::solvergraph::IFieldSpan<Tvec>,                                                   \
        spans_coordinates) /* center coordinates of each cell */                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_gravitational_force)

        EXPAND_NODE_EDGES(NODE_COMPUTE_ANALYTICAL_GRAVITY)

#undef NODE_COMPUTE_ANALYTICAL_GRAVITY

        void _impl_evaluate_internal();

        void _impl_reset_internal() {};

        inline virtual std::string _impl_get_label() const { return "ComputeAnalyticalGravity"; };

        virtual std::string _impl_get_tex() const { return "ComputeAnalyticalGravity"; };
    };

} // namespace shammodels::basegodunov::modules
