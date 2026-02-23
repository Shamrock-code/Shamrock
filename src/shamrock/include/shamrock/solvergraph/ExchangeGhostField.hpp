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
 * @file ExchangeGhostField.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solver graph node for exchanging ghost field data between distributed processes
 *
 * This file defines the ExchangeGhostField template class, which is a solver graph node responsible
 * for managing the communication of ghost field data across distributed computational
 * domains in the Shamrock hydrodynamics framework.
 */

#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief Solver graph node for distributed ghost field data exchange
     *
     * ExchangeGhostField is a templated computational node in the solver graph that handles
     * the communication and synchronization of ghost field data between distributed
     * computational domains. This node is essential for maintaining data consistency
     * at domain boundaries in parallel hydrodynamics simulations.
     *
     * The node operates by taking ownership rank information and performing sparse
     * communication to exchange typed patch data fields with appropriate neighboring
     * processes. It uses serialization/deserialization for efficient data transfer
     * and maintains shared distributed data structures for ghost zones.
     *
     * @tparam T The data type stored in the patch data fields to be exchanged
     *
     * @code{.cpp}
     * // Create the exchange node for double precision fields
     * auto exchangeNode = std::make_shared<ExchangeGhostField<double>>();
     *
     * // Set up edges with rank ownership and ghost field data
     * auto rankEdge = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>();
     * auto ghostEdge =
     * std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<double>>("ghost", "G");
     * exchangeNode->set_edges(rankEdge, ghostEdge);
     *
     * // Evaluate to perform the ghost field exchange
     * exchangeNode->evaluate();
     * @endcode
     */
    template<class T>
    class ExchangeGhostField : public shamrock::solvergraph::INode {

        shamalgs::collective::DDSCommCache cache;

        public:
        /**
         * @brief Default constructor for ExchangeGhostField node
         */
        ExchangeGhostField() {}

#define NODE_EXCHANGE_GHOST_FIELD_EDGES(X_RO, X_RW)                                                \
    /* input */                                                                                    \
    X_RO(shamrock::solvergraph::ScalarsEdge<u32>, rank_owner)                                      \
    /* output */                                                                                   \
    X_RW(shamrock::solvergraph::PatchDataFieldDDShared<T>, ghost_layer)

        EXPAND_NODE_EDGES(NODE_EXCHANGE_GHOST_FIELD_EDGES)
#undef NODE_EXCHANGE_GHOST_FIELD_EDGES

        /**
         * @brief Performs the ghost field data exchange computation
         *
         * This method implements the core functionality of the node by executing
         * sparse communication to exchange ghost field data between distributed
         * processes. It serializes local ghost data, communicates with appropriate
         * ranks based on ownership information, and deserializes received data.
         */
        void _impl_evaluate_internal();

        /**
         * @brief Returns the display label for this node
         * @return String label "ExchangeGhostField" for graph visualization
         */
        inline virtual std::string _impl_get_label() const { return "ExchangeGhostField"; };

        /**
         * @brief Returns the TeX representation for this node
         * @return TeX string for mathematical/graphical representation of the node
         */
        virtual std::string _impl_get_tex() const;
    };
} // namespace shamrock::solvergraph
