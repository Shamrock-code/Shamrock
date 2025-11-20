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
 * @file ComputeAMRLevel.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    class ComputeAMRLevel : public shamrock::solvergraph::INode {
        using Tgridscal = shambase::VecComponent<TgridVec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

        public:
        ComputeAMRLevel() {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::ScalarsEdge<TgridVec> &level0_size;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &block_min;
            const shamrock::solvergraph::IFieldSpan<TgridVec> &block_max;
            shamrock::solvergraph::IFieldSpan<TgridUint> &block_level;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<TgridVec>> level0_size,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> block_min,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridVec>> block_max,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<TgridUint>> block_level) {
            __internal_set_ro_edges({sizes, level0_size, block_min, block_max});
            __internal_set_rw_edges({block_level});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<TgridVec>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<TgridUint>>(0)};
        }

        void _impl_evaluate_internal();

        void _impl_reset_internal() {};

        inline virtual std::string _impl_get_label() const { return "ComputeAMRLevel"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
