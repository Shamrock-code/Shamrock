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
 * @file PatchDataLayerEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the PatchDataLayerEdge class for managing patch data layer edges.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include <memory>

namespace shamrock::solvergraph {

    class PatchDataLayerEdge : public IDataEdgeNamed {

        public:
        std::shared_ptr<patch::PatchDataLayerLayout> layout;
        shambase::DistributedData<patch::PatchDataLayer> patchdatas;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline PatchDataLayerEdge(
            std::string name,
            std::string label,
            std::shared_ptr<patch::PatchDataLayerLayout> layout)
            : IDataEdgeNamed(name, label), layout(layout) {}

        inline virtual const patch::PatchDataLayer &get(u64 id_patch) const {
            return patchdatas.get(id_patch);
        }

        inline virtual patch::PatchDataLayer &get(u64 id_patch) { return patchdatas.get(id_patch); }

        inline virtual void free_alloc() { patchdatas = {}; }
    };
} // namespace shamrock::solvergraph
