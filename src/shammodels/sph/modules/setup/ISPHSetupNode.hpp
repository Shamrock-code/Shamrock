// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ISPHSetupNode.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/patch/PatchData.hpp"
#include <string>
#include <vector>

namespace shammodels::sph::modules {

    struct ISPHSetupNode_Dot {
        std::string name;
        u32 type;
        std::vector<ISPHSetupNode_Dot> inputs;

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

    class ISPHSetupNode {
        public:
        virtual bool is_done() = 0;

        /**
         * @brief This function generate patchdata with at most nmax per MPI ranks
         * This function is always assumed as called by every ranks simultaneously
         *
         * @param nmax
         * @return shamrock::patch::PatchData
         */
        virtual shamrock::patch::PatchData next_n(u32 nmax) = 0;

        virtual std::string get_name() = 0;

        virtual ISPHSetupNode_Dot get_dot_subgraph() = 0;

        virtual ~ISPHSetupNode() = default;

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

    using SetupNodePtr = std::shared_ptr<ISPHSetupNode>;

} // namespace shammodels::sph::modules
