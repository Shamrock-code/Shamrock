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
 * @author Adnan-Ali Ahmad (adnan-ali.ahmad@cnrs.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)

 * @brief Computes the coordinates of each cell
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /********************* inputs *********************/                                           \
    /* number of blocks per patch for all patches on the current MPI process*/                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    /* min int coordinate of the block*/                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, spans_block_min)                             \
    /* max int coordinate of the block*/                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, spans_block_max)                             \
                                                                                                   \
    /********************* outputs *********************/                                          \
    /* center coordinates of each cell */                                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_coordinates)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeComputeCoordinates : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;

        u32 block_size;
        u32 block_nside;
        Tscal grid_coord_to_pos_fact;

        public:
        NodeComputeCoordinates(u32 block_size, u32 block_nside, Tscal grid_coord_to_pos_fact)
            : block_size(block_size), block_nside(block_nside),
              grid_coord_to_pos_fact(grid_coord_to_pos_fact) {

            if (block_nside != 2) {
                shambase::throw_with_loc<std::runtime_error>(
                    shambase::format("this module assume block_nside=2, got {}", block_nside));
            }
        }

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeCoordinates"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
