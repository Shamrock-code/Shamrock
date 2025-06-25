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
 * @file ScalarsEdge.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class ScalarsEdge : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        shambase::DistributedData<T> values;

        inline void free_alloc() {};
    };

} // namespace shamrock::solvergraph
