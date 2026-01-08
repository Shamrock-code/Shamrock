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
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include <memory>
#include <vector>

namespace shamrock::patch {
    /**
     * @brief PatchDataLayerLayout container class
     */
    class PatchDataLayout {
        public:
        std::vector<std::shared_ptr<shamrock::patch::PatchDataLayerLayout>> layer_layouts;
        size_t get_layer_count() const { return layer_layouts.size(); }

        inline std::shared_ptr<PatchDataLayerLayout> &get_layer_layout_ref(size_t idx = 0) {
            return shambase::get_check_ref(&layer_layouts.at(idx));
        }

        inline const std::shared_ptr<PatchDataLayerLayout> &get_layer_layout(size_t idx = 0) const {
            return layer_layouts.at(idx);
        }

        inline PatchDataLayerLayout &get_layer_layout_ptr(size_t idx = 0) {
            return *layer_layouts.at(idx);
        }

        inline void create_layers(size_t nlayers) {
            for (size_t idx = 0; idx < nlayers; idx++) {
                layer_layouts.push_back(std::make_shared<shamrock::patch::PatchDataLayerLayout>());
            }
        }
    };
} // namespace shamrock::patch
