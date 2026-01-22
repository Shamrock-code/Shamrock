// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MergedPatchDataEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SolverGraph edge for merged PatchDataLayer
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::gsph::solvergraph {

    /**
     * @brief SolverGraph edge for merged PatchDataLayer storage
     *
     * This edge stores distributed PatchDataLayer data that contains
     * merged local and ghost particle information. Used for:
     * - merged_xyzh: Position and smoothing length data for tree building
     * - merged_patchdata_ghost: Full field data including ghosts for force computation
     */
    class MergedPatchDataEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        shambase::DistributedData<shamrock::patch::PatchDataLayer> data;

        /**
         * @brief Get PatchDataLayer for a specific patch
         * @param id Patch ID
         */
        shamrock::patch::PatchDataLayer &get(u64 id) { return data.get(id); }

        /**
         * @brief Get PatchDataLayer for a specific patch (const)
         * @param id Patch ID
         */
        const shamrock::patch::PatchDataLayer &get(u64 id) const { return data.get(id); }

        /**
         * @brief Get the underlying distributed data
         */
        shambase::DistributedData<shamrock::patch::PatchDataLayer> &get_data() { return data; }

        /**
         * @brief Get the underlying distributed data (const)
         */
        const shambase::DistributedData<shamrock::patch::PatchDataLayer> &get_data() const {
            return data;
        }

        /**
         * @brief Free the allocated data
         */
        inline virtual void free_alloc() override { data = {}; }
    };

} // namespace shammodels::gsph::solvergraph
