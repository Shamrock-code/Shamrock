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
 * @file PatchDataLayout.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 */

#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"

namespace shamrock::patch {

    /**
     * @brief PatchData container class, made from a list of layers
     */
    class PatchDataLayout {
        std::vector<std::shared_ptr<PatchDataLayerLayout>> layouts;

        public:
        PatchDataLayout() {}

        PatchDataLayout(const std::vector<std::shared_ptr<PatchDataLayerLayout>> &layouts)
            : layouts(layouts) {}

        inline PatchDataLayerLayout &get_layer_layout(u32 idx) {
            return shambase::get_check_ref(layouts.at(idx));
        }

        inline const PatchDataLayerLayout &get_layer_layout(u32 idx) const {
            return shambase::get_check_ref(layouts.at(idx));
        }

        inline std::shared_ptr<PatchDataLayerLayout> &get_layer_layout_ptr(u32 idx) {
            return layouts.at(idx);
        }

        inline const std::shared_ptr<PatchDataLayerLayout> &get_layer_layout_ptr(u32 idx) const {
            return layouts.at(idx);
        }
    };

} // namespace shamrock::patch
