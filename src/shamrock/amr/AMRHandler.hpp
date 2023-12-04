// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRHandler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "AMRCell.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/patch/PatchData.hpp"

namespace shamrock::amr {

    template<class Tcoord, u32 dim, class AMRModel>
    class AMRHandler {

        PatchScheduler & sched;

        public:
        using CellCoord = AMRCellCoord<Tcoord, dim>;
        static constexpr u32 split_count = CellCoord::splts_count;

        void update_grid(){

            using namespace patch;

            //split

            sched.for_each_patch_data(
                [&](u64 id_patch, Patch cur_p, PatchData &pdat) {
                    
                    sycl::buffer<u32> split_list = AMRModel::get_split_table(pdat, cur_p);

                    

                }
            );


            //merge

        }

        
    };

} // namespace shamrock::amr