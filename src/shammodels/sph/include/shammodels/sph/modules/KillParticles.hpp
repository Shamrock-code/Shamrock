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
 * @file KillParticles.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the KillParticles module for removing particles.
 *
 */

#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"

namespace shammodels::sph::modules {

    class KillParticles : public shamrock::solvergraph::INode {

        public:
        KillParticles() {}

        struct Edges {
            const shamrock::solvergraph::DistributedBuffers<u32> &part_to_remove;
            shamrock::solvergraph::PatchDataLayerRefs &patchdatas;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::DistributedBuffers<u32>> part_to_remove,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> patchdatas) {
            __internal_set_ro_edges({part_to_remove});
            __internal_set_rw_edges({patchdatas});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::DistributedBuffers<u32>>(0),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerRefs>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "KillParticles"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::sph::modules
