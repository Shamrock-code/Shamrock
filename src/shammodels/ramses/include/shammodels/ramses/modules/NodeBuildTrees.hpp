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
 * @file NodeBuildTrees.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/solvegraph/TreeEdge.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtree/RadixTree.hpp"

namespace shammodels::basegodunov::modules {

    template<class Umorton, class TgridVec>
    class NodeBuildTrees : public shamrock::solvergraph::INode {

        u32 reduction_level = 0;

        using RTree = RadixTree<Umorton, TgridVec>;

        public:
        NodeBuildTrees() {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::FieldRefs<TgridVec> &block_min;
            const shamrock::solvergraph::FieldRefs<TgridVec> &block_max;
            solvergraph::TreeEdge<Umorton, TgridVec> &trees;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<TgridVec>> block_min,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<TgridVec>> block_max,
            std::shared_ptr<solvergraph::TreeEdge<Umorton, TgridVec>> trees) {
            __internal_set_ro_edges({sizes, block_min, block_max});
            __internal_set_rw_edges({trees});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::FieldRefs<TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::FieldRefs<TgridVec>>(2),
                get_rw_edge<solvergraph::TreeEdge<Umorton, TgridVec>>(0)};
        }

        void _impl_evaluate_internal();

        void _impl_reset_internal() {};

        inline virtual std::string _impl_get_label() const { return "BuildTrees"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
