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
 * @file ExchangeGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solver graph node for exchanging ghost layer data between distributed processes
 *
 * This file defines the ExchangeGhostLayer class, which is a solver graph node responsible
 * for managing the communication of ghost layer data across distributed computational
 * domains in the Shamrock hydrodynamics framework.
 */

#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief Solver graph node for distributed ghost layer data exchange
     *
     * ExchangeGhostLayer is a computational node in the solver graph that handles
     * the communication and synchronization of ghost layer data between distributed
     * computational domains. This node is essential for maintaining data consistency
     * at domain boundaries in parallel hydrodynamics simulations.
     *
     * The node operates by taking ownership rank information and performing sparse
     * communication to exchange patch data layers with appropriate neighboring
     * processes. It uses serialization/deserialization for efficient data transfer
     * and maintains shared distributed data structures for ghost zones.
     *
     * @code{.cpp}
     * // Create ghost layer layout
     * auto layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>(layout_params);
     *
     * // Create the exchange node
     * auto exchangeNode = std::make_shared<ExchangeGhostLayer>(layout);
     *
     * // Set up edges with rank ownership and ghost layer data
     * auto rankEdge = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>();
     * auto ghostEdge = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("ghost",
     * "G"); exchangeNode->set_edges(rankEdge, ghostEdge);
     *
     * // Evaluate to perform the ghost layer exchange
     * exchangeNode->evaluate();
     * @endcode
     */
    class ExchangeGhostLayer : public shamrock::solvergraph::INode {
        shamalgs::collective::DDSCommCache cache;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        /**
         * @brief Constructs an ExchangeGhostLayer node with specified layout
         *
         * @param ghost_layer_layout Shared pointer to the patch data layer layout
         *                          that defines the structure and organization of
         *                          ghost layer data to be exchanged
         */
        ExchangeGhostLayer(
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : ghost_layer_layout(std::move(ghost_layer_layout)) {}

#define NODE_EXCHANGE_GHOST_LAYER_EDGES(X_RO, X_RW)                                                \
    /* input */                                                                                    \
    X_RO(shamrock::solvergraph::ScalarsEdge<u32>, rank_owner)                                      \
    /* output */                                                                                   \
    X_RW(shamrock::solvergraph::PatchDataLayerDDShared, ghost_layer)

        EXPAND_NODE_EDGES(NODE_EXCHANGE_GHOST_LAYER_EDGES)
#undef NODE_EXCHANGE_GHOST_LAYER_EDGES

        /**
         * @brief Performs the ghost layer data exchange computation
         *
         * This method implements the core functionality of the node by executing
         * sparse communication to exchange ghost layer data between distributed
         * processes. It serializes local ghost data, communicates with appropriate
         * ranks based on ownership information, and deserializes received data.
         */
        void _impl_evaluate_internal();

        /**
         * @brief Returns the display label for this node
         * @return String label "ExchangeGhostLayer" for graph visualization
         */
        inline virtual std::string _impl_get_label() const { return "ExchangeGhostLayer"; };

        /**
         * @brief Returns the TeX representation for this node
         * @return TeX string for mathematical/graphical representation of the node
         */
        virtual std::string _impl_get_tex() const;
    };
} // namespace shamrock::solvergraph
