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
 * @file NodeUpdateDerivsCD10.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

#define NODE_UPDATE_DERIVS_CD10_EDGES(X_RO, X_RW)                                                  \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gpart_mass)                                     \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, alpha_u)                                        \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, beta_AV)                                        \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts_with_ghost)                              \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, xyz)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, vxyz)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, uint)                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, omega)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, pressure)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, cs)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, alpha_AV)                                       \
                                                                                                   \
    /* neigh */                                                                                    \
    X_RO(shammodels::sph::solvergraph::NeighCache, neigh_cache)                                    \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, axyz)                                            \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, duint)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class NodeUpdateDerivsCD10 : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;

        public:
        NodeUpdateDerivsCD10() {}

        EXPAND_NODE_EDGES(NODE_UPDATE_DERIVS_CD10_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "UpdateDerivsCD10"; };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::sph::modules
