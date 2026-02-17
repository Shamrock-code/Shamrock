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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Adnan-Ali Ahmad (adnan-ali.ahmad@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)

 * @brief Computes the coordinates of each cell
 *
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

#define NODE_COMPUTE_COORDINATES(X_RO, X_RW)                                                       \
    /* inputs */                                                                                   \
    X_RO(                                                                                          \
        shamrock::solvergraph::Indexes<u32>,                                                       \
        sizes) /* number of blocks per patch for all patches on the current MPI process*/          \
    X_RO(                                                                                          \
        shamrock::solvergraph::Field<Tscal>,                                                       \
        spans_block_cell_sizes) /* sizes of the cells within each block for all patches on the     \
                                   current MPI process*/                                           \
    X_RO(                                                                                          \
        shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>,                              \
        patch_boxes) /* bounding boxes of the patches  */                                          \
    X_RO(                                                                                          \
        shamrock::solvergraph::Field<Tvec>,                                                        \
        spans_cell0block_aabb_lower) /* coordinates of the lower left corner of each block */      \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(                                                                                          \
        shamrock::solvergraph::Field<Tvec>,                                                        \
        spans_coordinates) /* center coordinates of each cell */

        EXPAND_NODE_EDGES(NODE_COMPUTE_COORDINATES)

#undef NODE_COMPUTE_COORDINATES

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeCoordinates"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
