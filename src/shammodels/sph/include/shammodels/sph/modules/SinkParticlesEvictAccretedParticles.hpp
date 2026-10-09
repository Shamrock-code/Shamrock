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
 * @file SinkParticlesEvictAccretedParticles.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Remove SPH particles flagged for sink accretion from patch data.
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- Patchdatas ------------------- */                                       \
    X_RW(shamrock::solvergraph::PatchDataLayerRefs, pdats)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SinkParticlesEvictAccretedParticles : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkParticlesEvictAccretedParticles() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesEvictAccretedParticles";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
