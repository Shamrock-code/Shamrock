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
 * @file ISPHSetupNode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include <string>
#include <vector>

namespace shammodels::sph::modules {

    /**
     * @brief This struct is used to generate a dot graph of the setup tree
     *
     * It is composed of a name, a type and a vector of inputs.
     */
    struct ISPHSetupNode_Dot {
        std::string name;
        u32 type;
        std::vector<ISPHSetupNode_Dot> inputs;

        /**
         * @brief This function generate a dot graph for the setup tree
         *
         * This function is used to generate a dot graph that describes the
         * setup tree. It takes a counter and a string as input, and update
         * the counter and the string to generate the dot graph.
         *
         * @param counter a counter that is used to generate the node id in
         * the dot graph
         * @param out the string that will be updated to contain the dot graph
         * @return the new value of the counter
         */
        u32 add_node(u32 &counter, std::string &out) {

            std::vector<u32> inputs_id{};
            for (auto &in : inputs) {
                inputs_id.push_back(in.add_node(counter, out));
            }

            u32 counter_val = counter;
            counter++;

            out += "node_" + std::to_string(counter_val) + " [label=\"" + name + "\"];\n";

            for (auto i : inputs_id) {
                out += "node_" + std::to_string(i) + " -> node_" + std::to_string(counter_val)
                       + ";\n";
            }

            return counter_val;
        }
    };

    /**
     * @class ISPHSetupNode
     * @brief This class is an interface that all SPH setup nodes must implement.
     * It describe an operation associated to a node in the setup tree.
     */
    class ISPHSetupNode {
        public:
        /**
         * @brief This function return true if the setup is done
         *
         * @return true if done, false otherwise
         */
        virtual bool is_done() = 0;

        /**
         * @brief This function generate patchdata with at most nmax per MPI ranks
         * This function is always assumed as called by every ranks simultaneously
         *
         * @param nmax
         * @return shamrock::patch::PatchData
         */
        virtual shamrock::patch::PatchDataLayer next_n(u32 nmax) = 0;

        /**
         * @brief Get the name of the node
         * @return The name of the node
         */
        virtual std::string get_name() = 0;

        /**
         * @brief Get a dot subgraph describing the node and its childrens (recursively)
         *
         * This function should return a ISPHSetupNode_Dot object which contains
         * all the information needed to generate a dot graph for the node and
         * its children.
         *
         * @return A ISPHSetupNode_Dot object
         */
        virtual ISPHSetupNode_Dot get_dot_subgraph() = 0;

        /**
         * @brief Virtual destructor for the ISPHSetupNode class
         */
        virtual ~ISPHSetupNode() = default;

        /**
         * @brief Generate a dot graph for the setup tree
         *
         * This function returns a string containing a dot graph that describes the
         * setup tree.
         *
         * @return A string containing a dot graph
         */
        std::string get_dot() {
            std::string out;

            out += "digraph G {\n";
            out += "rankdir=LR;\n";

            u32 counter    = 0;
            u32 final_node = get_dot_subgraph().add_node(counter, out);

            out += "node_" + std::to_string(counter + 1) + " [label=\"Simulation\"];\n";
            out += "node_" + std::to_string(final_node) + " -> node_" + std::to_string(counter + 1)
                   + ";\n";

            out += "}\n";
            return out;
        }
    };

    /// Alias for a shared pointer to an ISPHSetupNode
    using SetupNodePtr = std::shared_ptr<ISPHSetupNode>;

} // namespace shammodels::sph::modules
