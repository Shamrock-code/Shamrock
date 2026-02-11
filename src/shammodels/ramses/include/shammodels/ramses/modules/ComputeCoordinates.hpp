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
 * @file ComputeCoordinates.hpp
 * @author
 * @brief Compute the coordinates of each cell from the coordinates of the lower left corner of each
 * block and the cell sizes
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeComputeCoordinates : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal grid_coord_to_pos_fact;

        public:
        NodeComputeCoordinates(u32 block_size, Tscal grid_coord_to_pos_fact)
            : block_size(block_size), grid_coord_to_pos_fact(grid_coord_to_pos_fact) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32>
                &sizes; // number of blocks per patch for all patches on the current MPI process
            const shamrock::solvergraph::Field<Tscal>
                &spans_block_cell_sizes; // sizes of the cells within each block for all patches on
                                         // the current MPI process
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>
                &patch_boxes; // bounding boxes of the patches
            const shamrock::solvergraph::Field<Tvec>
                &spans_cell0block_aabb_lower; // coordinates of the lower left corner of each block
            shamrock::solvergraph::Field<Tvec>
                &spans_coordinates; // center coordinates of each cell
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::Field<Tvec>> spans_cell0block_aabb_lower,
            std::shared_ptr<shamrock::solvergraph::Field<Tvec>> spans_coordinates) {
            __internal_set_ro_edges(
                {sizes, spans_block_cell_sizes, patch_boxes, spans_cell0block_aabb_lower});
            __internal_set_rw_edges({spans_coordinates});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(2),
                get_ro_edge<shamrock::solvergraph::Field<Tvec>>(3),
                get_rw_edge<shamrock::solvergraph::Field<Tvec>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeCoordinates"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
