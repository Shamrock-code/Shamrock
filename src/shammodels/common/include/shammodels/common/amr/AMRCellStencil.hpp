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
 * @file AMRCellStencil.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief utility to manipulate AMR blocks
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include <array>
#include <variant>
namespace shammodels::amr::cell {

    struct SameLevel {
        u64 cell_idx;
    };

    struct Levelp1 {
        u64 cell_child_idxs_base;

        template<class AMRBlock, class Fct>
        void for_all_indexes(Fct &&f) {

            f(cell_child_idxs_base + AMRBlock::get_index({0, 0, 0}));
            f(cell_child_idxs_base + AMRBlock::get_index({0, 0, 1}));
            f(cell_child_idxs_base + AMRBlock::get_index({0, 1, 0}));
            f(cell_child_idxs_base + AMRBlock::get_index({0, 1, 1}));
            f(cell_child_idxs_base + AMRBlock::get_index({1, 0, 0}));
            f(cell_child_idxs_base + AMRBlock::get_index({1, 0, 1}));
            f(cell_child_idxs_base + AMRBlock::get_index({1, 1, 0}));
            f(cell_child_idxs_base + AMRBlock::get_index({1, 1, 1}));
        }
    };

    struct Levelm1 {
        u64 cell_idx;
    };

    struct None {};

    /**
     * @brief Stencil element, describe the state of a cell relative to another
     *
     */
    struct alignas(8) StencilElement {

        enum { SAME, LEVELP1, LEVELM1, NONE } tag = NONE;

        union {
            SameLevel level_d0;
            Levelm1 level_dm1;
            Levelp1 level_dp1;
            None none;
        };

        static StencilElement make_none() {
            StencilElement ret;
            ret.tag  = NONE;
            ret.none = {};
            return ret;
        }
        static StencilElement make_same_level(SameLevel l) {
            StencilElement ret;
            ret.tag      = SAME;
            ret.level_d0 = l;
            return ret;
        }
        static StencilElement make_level_p1(Levelp1 l) {
            StencilElement ret;
            ret.tag       = LEVELP1;
            ret.level_dp1 = l;
            return ret;
        }
        static StencilElement make_level_m1(Levelm1 l) {
            StencilElement ret;
            ret.tag       = LEVELM1;
            ret.level_dm1 = l;
            return ret;
        }

        template<class Visitor1, class Visitor2, class Visitor3, class Visitor4>
        inline void visitor(Visitor1 &&f1, Visitor2 &&f2, Visitor3 &&f3, Visitor4 &&f4) {
            switch (tag) {
            case SAME   : f1(level_d0); break;
            case LEVELM1: f2(level_dm1); break;
            case LEVELP1: f3(level_dp1); break;
            case NONE   : f4(none); break;
            }
        }
    };

} // namespace shammodels::amr::cell
