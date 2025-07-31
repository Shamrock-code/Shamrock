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
 * @file InterfacesUtility.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamalgs/memory.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include <vector>

namespace shamrock {

    template<class T>
    class MergedPatchDataField {
        public:
        std::optional<shammath::CoordRange<T>> bounds;
        u32 original_elements;
        u32 total_elements;
        PatchDataField<T> field;

        bool has_bound_info() { return bounds.has_value(); }
    };

    class MergedPatchData {
        public:
        u32 original_elements;
        u32 total_elements;
        patch::PatchData pdat;
        patch::PatchDataLayout &pdl;
    };

    class InterfacesUtility {
        PatchScheduler &sched;

        public:
        InterfacesUtility(PatchScheduler &sched) : sched(sched) {}
    };
} // namespace shamrock
