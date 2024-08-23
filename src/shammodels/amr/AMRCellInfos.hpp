// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRCellInfos.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
