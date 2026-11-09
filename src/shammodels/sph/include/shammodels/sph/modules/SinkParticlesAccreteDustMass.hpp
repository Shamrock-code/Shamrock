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
 * @file SinkParticlesAccreteDustMass.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Track the dust mass, per dust bin, accreted onto sinks (monofluid TVA).
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
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, hfactd)                                          \
                                                                                                   \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
                                                                                                   \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- (sink) inputs ------------------- */                                    \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
                                                                                                   \
    /* ------------------- (sink) in/out ------------------- */                                    \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_dust_mass)

namespace shammodels::sph::modules {

    /**
     * @brief Add the dust mass of the particles flagged for accretion to their sink, per bin.
     *
     * The dust mass of bin j carried by particle a is m_a \epsilon_{j,a} with
     * \epsilon_{j,a} = S_{j,a}^2 / \rho_a. The sink_dust_mass edge holds one row of ndust bins
     * per sink : | bin0, bin1, ... | bin0, bin1, ... |.
     *
     * Must run after the accretion table is flagged and before the accreted particles are
     * evicted.
     */
    template<class Tvec>
    class SinkParticlesAccreteDustMass : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        SinkParticlesAccreteDustMass(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesAccreteDustMass";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
