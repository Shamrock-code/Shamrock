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
 * @file SchedulerPatchData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief PatchData handling

 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/patch/SimBox.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include <cmath>
#include <map>
#include <utility>

namespace shamrock::scheduler {

    using Patch             = shamrock::patch::Patch;
    using PatchData         = shamrock::patch::PatchData;
    using PatchDataLayout   = shamrock::patch::PatchDataLayout;
    using SimulationBoxInfo = shamrock::patch::SimulationBoxInfo;

    /**
     * @brief Class to handle PatchData owned by the node
     *
     */
    class SchedulerPatchData {
        public:
        PatchDataLayout &pdl;

        /**
         * @brief map container for patchdata owned by the current node (layout : id_patch,data)
         *
         */
        shambase::DistributedData<PatchData> owned_data;

        inline bool has_patch(u64 id) { return owned_data.has_key(id); }

        inline PatchData &get_pdat(u64 id) { return owned_data.get(id); }

        inline void for_each_patchdata(std::function<void(u64, PatchData &)> &&f) {
            owned_data.for_each(std::forward<std::function<void(u64, PatchData &)>>(f));
        }

        /**
         * @brief simulation box geometry info
         *
         */
        shamrock::patch::SimulationBoxInfo sim_box;

        /**
         * @brief apply a load balancing change list to shuffle patchdata arround the cluster
         *
         * @param change_list
         * @param patch_list
         */
        void apply_change_list(
            const shamrock::scheduler::LoadBalancingChangeList &change_list,
            SchedulerPatchList &patch_list);

        /**
         * @brief split a patchdata into 8 childs according to the 8 patches in arguments
         *
         * @param key_orginal key of the original patchdata
         * @param patches the patches
         */
        void split_patchdata(u64 key_orginal, const std::array<shamrock::patch::Patch, 8> patches);

        /**
         * @brief merge 8 old patchdata into one
         *
         * @param new_key new key to store the merge data in the map
         * @param old_keys old patch ids
         */
        void merge_patchdata(u64 new_key, const std::array<u64, 8> old_keys);

        inline SchedulerPatchData(
            shamrock::patch::PatchDataLayout &pdl, shamrock::patch::PatchCoord<3> patch_coord_range)
            : pdl(pdl), sim_box(pdl, patch_coord_range) {}
    };

} // namespace shamrock::scheduler
