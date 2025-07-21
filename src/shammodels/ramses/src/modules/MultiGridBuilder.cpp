// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MultiGridBuilder.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shammodels/ramses/modules/MultiGridBuilder.hpp"

namespace shammodels::basegodunov::modules {
    template<class TgridVec>
    void BuildMultigrid<TgridVec>::_impl_evaluate_internal() {

        auto edges = get_edges();

        // get histogram of cell sizes
        // aka: counts = [count for i in local_levels]

        // compute exscan offsets for tree levels
        // aka: offsets = [offset_start_level for i in local_levels]
        //              = exclusive_scan(counts)

        // compute replication count for all cells
        // aka: replicate_count = cell_level - root_level
    }

    template<class TgridVec>
    std::string BuildMultigrid<TgridVec>::_impl_get_tex() {
        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::BuildMultigrid<i64_3>;
