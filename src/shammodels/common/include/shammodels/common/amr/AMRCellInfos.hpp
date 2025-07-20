// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRCellInfos.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambackends/sycl.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    struct CellInfos {

        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        // size of a cell of a block = block_cell_sizes[block]
        shamrock::ComputeField<Tscal> block_cell_sizes;

        // the center of the first cell in the block
        // cell0block_aabb[block] + lcoord[loc_id]*block_cell_sizes[block] = cell0 aabb [global cell
        // id]
        shamrock::ComputeField<Tvec> cell0block_aabb_lower;

        // upper is not needed since it is cell0block_aabb_lower + block_cell_sizes
    };

} // namespace shammodels::basegodunov::modules
