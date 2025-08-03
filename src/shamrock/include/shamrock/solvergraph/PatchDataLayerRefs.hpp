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
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include <functional>

namespace shamrock::solvergraph {

    class PatchDataLayerRefs : public IDataEdgeNamed {

        public:
        shambase::DistributedData<std::reference_wrapper<patch::PatchDataLayer>> patchdatas;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline virtual patch::PatchDataLayer &get(u64 id_patch) const {
            return patchdatas.get(id_patch);
        }

        inline virtual void free_alloc() { patchdatas = {}; }
    };
} // namespace shamrock::solvergraph
