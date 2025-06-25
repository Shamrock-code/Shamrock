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
 * @file ComputeCubeCellSizes.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <type_traits>

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    class NodeComputeCubeCellSizes : public shamrock::solvergraph::INode {
        using TgridScal = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

        public:
        NodeComputeCubeCellSizes() {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &spans_block_min;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &spans_block_max;
            shamrock::solvergraph::IFieldSpan<TgridScal> &spans_block_cell_sizes;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> spans_block_min,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> spans_block_max,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridScal>> spans_block_cell_sizes) {
            __internal_set_ro_edges({sizes, spans_block_min, spans_block_max});
            __internal_set_rw_edges({spans_block_cell_sizes});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<TgridScal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeCubeCellSizes"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
