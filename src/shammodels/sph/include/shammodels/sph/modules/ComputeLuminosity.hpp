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
 * @file ComputeLuminosity.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts_with_ghosts)                             \
    X_RO(shammodels::sph::solvergraph::NeighCache, neigh_cache)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, xyz)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, omega)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, u)                                              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, pressure)                                       \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, luminosity)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class NodeComputeLuminosity : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal part_mass;
        Tscal alpha_u;

        public:
        NodeComputeLuminosity(Tscal part_mass, Tscal alpha_u)
            : part_mass(part_mass), alpha_u(alpha_u) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeLuminosity"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
