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
 * @file SinkParticlesFlagAccreteHard.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Flag SPH particles inside sink accretion radii into an accretion table.
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/IFreeable.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
                                                                                                   \
    /* ------------------- (sink) inputs ------------------- */                                    \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_accr_radii)                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    /* sink_accretion_table[id_a] = who should accrete part [id_a] (or u32_max if none); */        \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SinkParticlesFlagAccreteHard : public shamrock::solvergraph::INode,
                                         public shamrock::solvergraph::IFreeable {

        using Tscal = shambase::VecComponent<Tvec>;

        std::unique_ptr<sham::DeviceBuffer<Tvec>> sink_pos;
        std::unique_ptr<sham::DeviceBuffer<Tscal>> sink_accr_radii;

        public:
        SinkParticlesFlagAccreteHard() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline void free_alloc() {
            sink_pos        = {};
            sink_accr_radii = {};
        }

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesFlagAccreteHard";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
