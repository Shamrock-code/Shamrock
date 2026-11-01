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
 * @file ComputeNeighStats.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief A module to compute and display statistics on neighbor counts for SPH particles.
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shammodels::sph::solvergraph::NeighCache, neigh_cache)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, xyz)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)

namespace shammodels::sph::modules {

    template<class Tvec>
    class ComputeNeighStats : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal kernel_radius;

        public:
        ComputeNeighStats(Tscal kernel_radius) : kernel_radius(kernel_radius) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeNeighStats"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
