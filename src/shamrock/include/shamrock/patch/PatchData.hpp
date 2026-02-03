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
 * @file PatchData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 */

#include "shamrock/patch/PatchDataLayer.hpp"

namespace shamrock::patch {

    /**
     * @brief PatchData container class, made from a list of layers
     */
    class PatchData {
        std::vector<std::shared_ptr<PatchDataLayer>> layers;

        public:
        PatchData() {}

        PatchData(const std::vector<std::shared_ptr<PatchDataLayer>> &layers) : layers(layers) {}

        inline PatchDataLayer &get_layer(u32 idx) {
            return shambase::get_check_ref(layers.at(idx));
        }

        inline const PatchDataLayer &get_layer(u32 idx) const {
            return shambase::get_check_ref(layers.at(idx));
        }

        inline std::shared_ptr<PatchDataLayer> &get_layer_ptr(u32 idx) { return layers.at(idx); }

        inline const std::shared_ptr<PatchDataLayer> &get_layer_ptr(u32 idx) const {
            return layers.at(idx);
        }
    };

} // namespace shamrock::patch
