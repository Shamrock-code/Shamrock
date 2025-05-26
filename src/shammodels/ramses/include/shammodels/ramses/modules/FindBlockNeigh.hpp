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
 * @file FindBlockNeigh.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shammodels/ramses/solvegraph/TreeEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeigh : public shamrock::solvergraph::INode {
        using Tscal            = shambase::VecComponent<Tvec>;
        using RTree            = RadixTree<Tmorton, TgridVec>;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        class AMRBlockFinder;

        public:
        FindBlockNeigh() {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldRefs<TgridVec> &spans_block_min;
            const shamrock::solvergraph::IFieldRefs<TgridVec> &spans_block_max;
            const solvergraph::TreeEdge<Tmorton, TgridVec> &trees;
            solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &block_neigh_graph;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<TgridVec>> spans_block_min,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<TgridVec>> spans_block_max,
            std::shared_ptr<solvergraph::TreeEdge<Tmorton, TgridVec>> trees,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> block_neigh_graph) {
            __internal_set_ro_edges({sizes, spans_block_min, spans_block_max, trees});
            __internal_set_rw_edges({block_neigh_graph});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<TgridVec>>(2),
                get_ro_edge<solvergraph::TreeEdge<Tmorton, TgridVec>>(3),
                get_rw_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "FindBlockNeigh"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
