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
 * @file FindGhostLayerCandidates.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"

namespace shammodels::basegodunov::modules {

    enum class GhostType { None, Periodic, Reflective };

    struct GhostLayerGenMode {
        GhostType ghost_type_x;
        GhostType ghost_type_y;
        GhostType ghost_type_z;
    };

    template<class TgridVec>
    shammath::paving_function_general_3d<TgridVec>
    get_paving(GhostLayerGenMode mode, shammath::AABB<TgridVec> sim_box) {

        TgridVec box_size   = sim_box.upper - sim_box.lower;
        TgridVec box_center = (sim_box.upper + sim_box.lower) / 2;

        SHAM_ASSERT(sim_box.is_volume_not_null());

        { // check that rebuildind the AABB from size and center gives the same AABB
            shammath::AABB<TgridVec> new_box
                = {box_center - box_size / 2, box_center + box_size / 2};
            if (new_box != sim_box) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Rebuilding AABB from size and center gives a different AABB");
            }
        }

        return shammath::paving_function_general_3d<TgridVec>{
            box_size,
            box_center,
            mode.ghost_type_x == GhostType::Periodic,
            mode.ghost_type_y == GhostType::Periodic,
            mode.ghost_type_z == GhostType::Periodic};
    }

} // namespace shammodels::basegodunov::modules
