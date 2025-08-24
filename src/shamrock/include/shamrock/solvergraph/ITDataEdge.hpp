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
 * @file ITDataEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include <string>
#include <utility>

namespace shamrock::solvergraph {

    template<class T>
    class ITDataEdge : public IDataEdgeNamed {

        public:
        T data;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline virtual void free_alloc() { data = {}; }

        virtual ~ITDataEdge() {}
    };

} // namespace shamrock::solvergraph
