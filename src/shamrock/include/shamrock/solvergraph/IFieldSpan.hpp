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
 * @file IFieldSpan.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class IFieldSpan : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        virtual shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>>
        get_spans() const = 0;

        inline virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const = 0;

        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) = 0;
    };

} // namespace shamrock::solvergraph
