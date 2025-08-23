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
 * @file PatchDataFieldDDShared.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/DistributedDataShared.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class PatchDataFieldDDShared : public IDataEdgeNamed {

        public:
        shambase::DistributedDataShared<PatchDataField<T>> patchdata_fields;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline virtual void free_alloc() { patchdata_fields = {}; }
    };

} // namespace shamrock::solvergraph
