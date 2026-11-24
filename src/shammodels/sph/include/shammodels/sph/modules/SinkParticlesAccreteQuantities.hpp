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

#include "shambase/exception.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <stdexcept>
#include <vector>

#define NODE_EDGES(X_RO, X_RW, X_RO_OPTIONAL, X_RW_OPTIONAL)                                       \
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
    /* ------------------- (field) dust inputs (monofluid TVA) ------------------- */              \
    /* S_j = sqrt(rho epsilon_j), needs h to get rho */                                            \
    X_RO_OPTIONAL(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                 \
    X_RO_OPTIONAL(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                   \
                                                                                                   \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- (sink) in/out ------------------- */                                    \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_velocities)                     \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_accelerations)                  \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_angmom)                         \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_mass)                          \
    X_RW_OPTIONAL(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_accreted_dust_mass)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SinkParticlesAccreteQuantities : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;
        Tscal hfactd;

        public:
        /**
         * @param ndust number of dust species whose accreted mass is tracked in
         * sink_accreted_dust_mass (0 to disable the tracking)
         * @param hfactd SPH kernel hfactd, used to recover rho from h with S_j (required when
         * ndust != 0)
         */
        SinkParticlesAccreteQuantities(u32 ndust = 0, Tscal hfactd = 0)
            : ndust(ndust), hfactd(hfactd) {
            if (ndust != 0 && hfactd == 0) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "hfactd must be set when tracking the accreted dust mass (ndust != 0)");
            }
        }

        EXPAND_NODE_EDGES_OPTIONAL(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesAccreteQuantities";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
