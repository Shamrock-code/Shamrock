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
 * @file BuildGhostInterfaceIdTable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solvergraph node selecting the ids of the particles sent through each ghost interface
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::DDSharedScalar<InterfaceBuildInfos>, interface_infos)              \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::DDSharedScalar<InterfaceIdTable>, interface_id_table)

namespace shammodels::sph::modules {

    /**
     * @brief Build the id table of every ghost interface
     *
     * For each interface (sender -> receiver) described in `interface_infos`, select the ids of
     * the particles of the sender patch that lie within the interface cut volume. Interfaces
     * that end up empty are dropped, as are interfaces whose sender is not in `positions` (which
     * only holds non-empty patches). A warning is emitted if the ghost volume of a patch is too
     * large compared to the patch itself.
     *
     * @tparam Tvec position vector type
     */
    template<class Tvec>
    class BuildGhostInterfaceIdTable : public shamrock::solvergraph::INode {

        using GhostHandle         = BasicSPHGhostHandler<Tvec>;
        using InterfaceBuildInfos = typename GhostHandle::InterfaceBuildInfos;
        using InterfaceIdTable    = typename GhostHandle::InterfaceIdTable;

        public:
        BuildGhostInterfaceIdTable() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "BuildGhostInterfaceIdTable"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
