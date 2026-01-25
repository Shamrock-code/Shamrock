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
 * @file MergedPatchDataEdge.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SolverGraph edge for merged PatchDataLayer
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shammodels::gsph::solvergraph {

    /// SolverGraph edge for merged PatchDataLayer storage (local + ghost particles)
    class MergedPatchDataEdge : public shamrock::solvergraph::IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        shambase::DistributedData<shamrock::patch::PatchDataLayer> data;

        shamrock::patch::PatchDataLayer &get(u64 id) { return data.get(id); }
        const shamrock::patch::PatchDataLayer &get(u64 id) const { return data.get(id); }

        shambase::DistributedData<shamrock::patch::PatchDataLayer> &get_data() { return data; }
        const shambase::DistributedData<shamrock::patch::PatchDataLayer> &get_data() const {
            return data;
        }

        inline virtual void free_alloc() override { data = {}; }
    };

} // namespace shammodels::gsph::solvergraph
