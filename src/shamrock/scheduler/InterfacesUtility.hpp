// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file InterfacesUtility.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamalgs/memory.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <vector>

namespace shamrock {

    template<class T>
    class MergedPatchDataField{public:
        std::optional<shammath::CoordRange<T>> bounds;
        u32 original_elements;
        u32 total_elements;
        PatchDataField<T> field;

        bool has_bound_info(){
            return bounds.has_value();
        }
    };

    class MergedPatchData{public:
    
        u32 original_elements;
        u32 total_elements;
        patch::PatchData pdat;
        patch::PatchDataLayout & pdl;

    };



    class InterfacesUtility {
        PatchScheduler &sched;

        public:
        InterfacesUtility(PatchScheduler &sched) : sched(sched) {}

        
    };
} // namespace shamrock