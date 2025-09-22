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
 * @file PatchDataLayerRefs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the PatchDataLayerRefs class for managing distributed references to patch data
 * layers.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"

namespace shamrock::solvergraph {

    class PatchDataLayerRefs : public IPatchDataLayerRefs {

        public:
        shambase::DistributedData<PatchDataLayerRef> patchdatas;

        using IPatchDataLayerRefs::IPatchDataLayerRefs;

        inline virtual patch::PatchDataLayer &get(u64 id_patch) override {
            return patchdatas.get(id_patch);
        }

        inline virtual const patch::PatchDataLayer &get(u64 id_patch) const override {
            return patchdatas.get(id_patch);
        }

        inline virtual shambase::DistributedData<PatchDataLayerRef> &get_refs() override {
            return patchdatas;
        }

        inline virtual const shambase::DistributedData<PatchDataLayerRef> &get_const_refs()
            const override {
            return patchdatas;
        }

        inline virtual void free_alloc() override { patchdatas = {}; }
    };

} // namespace shamrock::solvergraph
