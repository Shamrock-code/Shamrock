// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

        public:
        /**
         * @brief Default constructor for ExchangeGhostField node
         */
        ExchangeGhostField() {}

        /**
         * @brief Structure containing references to input and output data edges
         *
         * This structure provides convenient access to the solver graph edges
         * that this node operates on. It encapsulates both read-only input data
         * and read-write output data used during ghost field exchange.
         *
         * @var rank_owner Read-only edge containing rank ownership information for each patch ID
         * @var ghost_layer Read-write edge containing shared distributed ghost field data
         */
        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<u32> &rank_owner;
            shamrock::solvergraph::PatchDataFieldDDShared<T> &ghost_layer;
        };

        /**
         * @brief Sets the input and output data edges for this node
         *
         * This method connects the node to its required data dependencies in the
         * solver graph. It establishes the read-only connection to rank ownership
         * data and the read-write connection to ghost field data.
         *
         * @param rank_owner Shared pointer to edge containing rank ownership mapping for patch IDs
         * @param ghost_layer Shared pointer to edge containing distributed ghost field data to be
         * exchanged
         */
        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> rank_owner,
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> ghost_layer) {
            __internal_set_ro_edges({rank_owner});
            __internal_set_rw_edges({ghost_layer});
        }

        /**
         * @brief Retrieves references to the connected data edges
         *
         * @return Edges structure containing references to the rank ownership
         *         and ghost field data edges used by this node
         */
        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<u32>>(0),
                get_rw_edge<shamrock::solvergraph::PatchDataFieldDDShared<T>>(0),
            };
        }

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
