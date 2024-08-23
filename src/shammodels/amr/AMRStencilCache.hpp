// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRStencilCache.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/amr/AMRBlockStencil.hpp"
#include "shammodels/amr/AMRCellStencil.hpp"

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
