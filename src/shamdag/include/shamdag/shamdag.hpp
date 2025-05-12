// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file shamdag.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <functional>
#include <memory>
#include <vector>

class INode;
class IDataEdge;

/**
 * Evaluates a function on all nodes in the graph below the given endpoints in a depth-first manner.
 * Nodes are evaluated if the condition evaluates to true and has not been evaluated before.
 *
 * @note This function will evaluate in the correct order for the DAG (deep-first).
 *
 * @param endpoints starting nodes
 * @param f function to evaluate
 * @param condition condition for nodes to be evaluated
 */
void multi_evaluate(
    std::vector<std::shared_ptr<INode>> endpoints,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition);

/**
 * Evaluates a function on all nodes in the graph above the given entrypoint. The order is the
 * opposite of the one in multi_evaluate.
 *
 * @note The order is the opposite of the one in multi_evaluate. As a result allocations can be
 * free'd in reverse order without a clash.
 *
 * @param endpoints starting nodes
 * @param f function to evaluate
 * @param condition condition for nodes to be evaluated
 */
void multi_evaluate_up(
    std::vector<std::shared_ptr<INode>> entrypoint,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition);

/**
 * @brief Filters out duplicate nodes from a vector of nodes
 *
 * @param nodes vector of nodes
 * @return vector of nodes with duplicates removed
 */
std::vector<std::shared_ptr<INode>> filter_duplicate(std::vector<std::shared_ptr<INode>> nodes);

/**
 * @brief Retrieves all endpoint nodes in the subgraph given by the given nodes.
 *
 * @param nodes A vector of nodes from which to gather the endpoints.
 * @return A vector of endpoint nodes, with duplicates filtered out.
 *
 * Endpoints are nodes that have no outputs, indicating termination points
 * in a directed acyclic graph (DAG).
 */
std::vector<std::shared_ptr<INode>> get_endpoints(std::vector<std::shared_ptr<INode>> nodes);

/**
 * @brief Retrieves all entrypoint nodes in the subgraph given by the given nodes.
 *
 * @param nodes A vector of nodes from which to gather the entrypoints.
 * @return A vector of entrypoint nodes, with duplicates filtered out.
 *
 * Entrypoints are nodes that have no inputs, indicating the starting points
 * in a directed acyclic graph (DAG).
 */
std::vector<std::shared_ptr<INode>> get_entrypoints(std::vector<std::shared_ptr<INode>> nodes);
