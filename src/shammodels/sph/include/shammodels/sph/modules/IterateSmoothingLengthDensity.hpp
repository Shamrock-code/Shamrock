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
 * @file IterateSmoothingLengthDensity.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the IterateSmoothingLengthDensity module for iterating smoothing length based on
 * the SPH density sum.
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shammodels::sph::solvergraph::NeighCache, neigh_cache)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, old_h)                                          \
                                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, new_h)                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, eps_h)

namespace shammodels::sph::modules {

    template<class Tvec, class SPHKernel>
    class IterateSmoothingLengthDensity : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal gpart_mass;
        Tscal h_evol_max;
        Tscal h_evol_iter_max;
        Tscal epsilon_h;

        public:
        IterateSmoothingLengthDensity(
            Tscal gpart_mass, Tscal h_evol_max, Tscal h_evol_iter_max, Tscal epsilon_h)
            : gpart_mass(gpart_mass), h_evol_max(h_evol_max), h_evol_iter_max(h_evol_iter_max),
              epsilon_h(epsilon_h) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "IterateSmoothingLengthDensity";
        };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
