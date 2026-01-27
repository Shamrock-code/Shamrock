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
 * @file InterpolationUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"

namespace {

    template<class Tvec, class TgridVec, class AMRBlock>
    class GetShift {

        public:
        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        const Tvec *acc_aabb_block_lower;
        const Tscal *acc_aabb_cell_size;

        GetShift(const Tvec *aabb_block_lower, const Tscal *aabb_cell_size)
            : acc_aabb_block_lower{aabb_block_lower}, acc_aabb_cell_size{aabb_cell_size} {}

        shammath::AABB<Tvec> get_cell_aabb(u32 id) const {

            const u32 cell_global_id = (u32) id;

            const u32 block_id    = cell_global_id / AMRBlock::block_size;
            const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

            // fetch current block info
            const Tvec cblock_min  = acc_aabb_block_lower[block_id];
            const Tscal delta_cell = acc_aabb_cell_size[block_id];

            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
            Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

            Tvec aabb_min = cblock_min + offset;
            Tvec aabb_max = aabb_min + delta_cell;

            return {aabb_min, aabb_max};
        }

        std::pair<Tvec, Tvec> get_shifts(u32 id_a, u32 id_b) const {

            shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
            shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

            shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

            Tvec face_center = face_aabb.get_center();

            Tvec shift_a = face_center - aabb_cell_a.get_center();
            Tvec shift_b = face_center - aabb_cell_b.get_center();

            return {shift_a, shift_b};
        }
    };
} // namespace
