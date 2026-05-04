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
 * @file RankGetter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <functional>
#include <utility>

namespace shamrock::solvergraph {

    /// Edge to get the rank owning a patch. Could be made faster by usage of a cache, provided that
    /// we ensure its reset
    class RankGetter : public IEdgeNamed {

        /// internal getter function passed on construction
        std::function<u32(u64)> rank_getter_func;

        public:
        RankGetter(std::function<u32(u64)> getter_func, std::string label, std::string tex_symbol)
            : rank_getter_func(std::move(getter_func)),
              IEdgeNamed(std::move(label), std::move(tex_symbol)) {}

        inline u32 get_rank_owner(u64 patch_id) const { return rank_getter_func(patch_id); }

        inline void free_alloc() {};
    };

} // namespace shamrock::solvergraph
