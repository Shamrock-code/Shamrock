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
 * @file AMRStencilCache.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/common/amr/AMRBlockStencil.hpp"
#include "shammodels/common/amr/AMRCellStencil.hpp"

namespace shammodels::basegodunov {

    struct AMRStencilCache {

        using cell_stencil_el_buf = std::unique_ptr<sycl::buffer<amr::cell::StencilElement>>;

        using dd_cell_stencil_el_buf = shambase::DistributedData<cell_stencil_el_buf>;
        std::unordered_map<u32, dd_cell_stencil_el_buf> storage;

        void insert_data(u32 map_id, dd_cell_stencil_el_buf &&stencil_element) {
            storage.emplace(map_id, std::forward<dd_cell_stencil_el_buf>(stencil_element));
        }

        sycl::buffer<amr::cell::StencilElement> &get_stencil_element(u64 patch_id, u32 map_id) {
            return shambase::get_check_ref(storage.at(map_id).get(patch_id));
        }
    };

} // namespace shammodels::basegodunov
