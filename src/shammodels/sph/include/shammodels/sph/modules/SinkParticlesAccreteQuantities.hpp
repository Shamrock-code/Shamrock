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
 * @file SinkParticlesAccreteQuantities.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Accrete flagged SPH particles onto sinks (mass, CoM, spin, etc.).
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (param) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
                                                                                                   \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, velocities)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, accelerations)                                   \
                                                                                                   \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- (sink) in/out ------------------- */                                    \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_velocities)                     \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_accelerations)                  \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_angmom)                         \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_mass)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SinkParticlesAccreteQuantities : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkParticlesAccreteQuantities() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesAccreteQuantities";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
