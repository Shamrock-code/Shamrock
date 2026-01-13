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
 * @file ScalarEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class ScalarEdge : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        T value;

        inline void free_alloc() {};
    };

} // namespace shamrock::solvergraph
