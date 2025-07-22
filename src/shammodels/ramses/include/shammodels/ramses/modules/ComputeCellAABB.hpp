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
 * @file ComputeCellAABB.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeComputeCellAABB : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_nside;
        Tscal grid_coord_to_pos_fact;

        public:
        NodeComputeCellAABB(u32 block_nside, Tscal grid_coord_to_pos_fact)
            : block_nside(block_nside), grid_coord_to_pos_fact(grid_coord_to_pos_fact) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &spans_block_min;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &spans_block_max;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_cell0block_aabb_lower;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> spans_block_min,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> spans_block_max,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_cell0block_aabb_lower) {
            __internal_set_ro_edges({sizes, spans_block_min, spans_block_max});
            __internal_set_rw_edges({spans_block_cell_sizes, spans_cell0block_aabb_lower});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(1),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ComputeCellAABB"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
