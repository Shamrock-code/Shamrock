// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRBlockCellLowering.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief utility to manipulate AMR blocks
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/common/amr/AMRBlockStencil.hpp"
#include "shammodels/common/amr/AMRCellStencil.hpp"
#include <array>
#include <variant>

namespace shammodels::amr {

    enum StencilPosition {
        xp1 = 0,
        xm1 = 1,
        yp1 = 2,
        ym1 = 3,
        zp1 = 4,
        zm1 = 5,
    };

    struct BlockCellLowering {

        static constexpr u32 dim = 3;
        using AMRBlock           = amr::AMRBlock<f64_3, i64_3, 2>;

        template<StencilPosition stencil_pos>
        inline static constexpr std::array<i32, 3> get_relative_offset_vec() {
            if constexpr (stencil_pos == xp1) {
                return std::array<i32, 3>{1, 0, 0};
            } else if constexpr (stencil_pos == xm1) {
                return std::array<i32, 3>{-1, 0, 0};
            } else if constexpr (stencil_pos == yp1) {
                return std::array<i32, 3>{0, 1, 0};
            } else if constexpr (stencil_pos == ym1) {
                return std::array<i32, 3>{0, -1, 0};
            } else if constexpr (stencil_pos == zp1) {
                return std::array<i32, 3>{0, 0, 1};
            } else if constexpr (stencil_pos == zm1) {
                return std::array<i32, 3>{0, 0, -1};
            } else {
                static_assert(
                    shambase::always_false_v<decltype(stencil_pos)>, "non-exhaustive visitor!");
            }
        }

        template<StencilPosition stencil_pos>
        inline static constexpr i32 get_relative_offset() {
            return AMRBlock::get_index_relative(get_relative_offset_vec<stencil_pos>());
        }

        template<StencilPosition stencil_pos>
        inline static constexpr cell::StencilElement lower_block_to_cell(
            u32 block_id,
            u32 local_cell_id,
            std::array<u32, dim> lcoord,
            block::StencilElement block_stencil_el) {

            // check if still within block (and return if thats the case)

            i32 offset_lcoord = i32(local_cell_id) + get_relative_offset<stencil_pos>();

            if (offset_lcoord > 0 && offset_lcoord < AMRBlock::block_size) {
                return cell::StencilElement{
                    cell::SameLevel{block_id * AMRBlock::block_size + offset_lcoord}};
            }

            // the previous part has not returned so we have to look at the neighbouring block

            constexpr std::array<i32, 3> off = get_relative_offset_vec<stencil_pos>();

            // offset local coordinates
            std::array<i32, dim> coord_off = {
                lcoord[0] + off[0],
                lcoord[1] + off[1],
                lcoord[2] + off[2],
            };

            // make the local coord relative to the neighbouring block
            if constexpr (stencil_pos == xp1) {
                coord_off[0] -= AMRBlock::side_size;
            } else if constexpr (stencil_pos == xm1) {
                coord_off[0] += AMRBlock::side_size;
            } else if constexpr (stencil_pos == yp1) {
                coord_off[1] -= AMRBlock::side_size;
            } else if constexpr (stencil_pos == ym1) {
                coord_off[1] += AMRBlock::side_size;
            } else if constexpr (stencil_pos == zp1) {
                coord_off[2] -= AMRBlock::side_size;
            } else if constexpr (stencil_pos == zm1) {
                coord_off[2] += AMRBlock::side_size;
            } else {
                static_assert(
                    shambase::always_false_v<decltype(stencil_pos)>, "non-exhaustive visitor!");
            }

            return block_stencil_el.visitor_ret<cell::StencilElement>(
                [&](block::SameLevel st) {
                    return cell::SameLevel{
                        st.block_idx * AMRBlock::block_size
                        + AMRBlock::get_index(
                            {u32(coord_off[0]), u32(coord_off[1]), u32(coord_off[2])})};
                },
                [&](block::Levelm1 st) {
                    std::array<u32, 3> mod_coord{
                        u32(coord_off[0]) % 2,
                        u32(coord_off[1]) % 2,
                        u32(coord_off[2]) % 2,
                    };

                    std::array<i32, 3> block_pos_offset{
                        (0b100 & st.neighbourgh_state) >> 2,
                        (0b010 & st.neighbourgh_state) >> 1,
                        (0b001 & st.neighbourgh_state) >> 0,
                    };

                    block_pos_offset[0] *= AMRBlock::side_size / 2;
                    block_pos_offset[1] *= AMRBlock::side_size / 2;
                    block_pos_offset[2] *= AMRBlock::side_size / 2;

                    coord_off = {
                        block_pos_offset[0] + (coord_off[0] / 2),
                        block_pos_offset[1] + (coord_off[1] / 2),
                        block_pos_offset[2] + (coord_off[2] / 2),
                    };

                    auto neigh_state
                        = cell::Levelm1::STATE(mod_coord[0] * 4 + mod_coord[1] * 2 + mod_coord[2]);

                    return cell::Levelm1{
                        neigh_state,
                        st.block_idx * AMRBlock::block_size
                            + AMRBlock::get_index(
                                {u32(coord_off[0]), u32(coord_off[1]), u32(coord_off[2])})};
                },
                [&](block::Levelp1 st) {
                    std::array<u32, 3> mod_coord{
                        u32(coord_off[0] / (AMRBlock::side_size / 2)),
                        u32(coord_off[1] / (AMRBlock::side_size / 2)),
                        u32(coord_off[2] / (AMRBlock::side_size / 2)),
                    };
                    u32 child_select = mod_coord[0] * 4 + mod_coord[1] * 2 + mod_coord[2];

                    coord_off
                        = {int((coord_off[0] * 2) % AMRBlock::side_size),
                           int((coord_off[1] * 2) % AMRBlock::side_size),
                           int((coord_off[2] * 2) % AMRBlock::side_size)};

                    u32 idx_tmp = st.block_child_idxs[child_select] * AMRBlock::block_size
                                  + AMRBlock::get_index(
                                      {u32(coord_off[0]), u32(coord_off[1]), u32(coord_off[2])});

                    return cell::Levelp1{idx_tmp};
                },
                [&](block::StencilElement::None st) {
                    return cell::StencilElement::None{};
                });
        }
    };

} // namespace shammodels::amr
