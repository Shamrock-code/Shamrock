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
 * @file NodeAXPYThreeVectors.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeAXPYThreeVectors : public shamrock::solvergraph::INode {

        u32 block_size;

        public:
        NodeAXPYThreeVectors(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<T> &spans_x;
            const shamrock::solvergraph::IFieldSpan<T> &spans_y;
            const shamrock::solvergraph::ScalarEdge<T> &alpha;
            const shamrock::solvergraph::ScalarEdge<T> &beta;
            shamrock::solvergraph::IFieldRefs<T> &spans_z;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> spans_x,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> spans_y,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> alpha,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> beta,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> spans_z) {
            __internal_set_ro_edges({sizes, spans_x, spans_x, alpha, beta});
            __internal_set_rw_edges({spans_z});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<T>>(3),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<T>>(4),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<T>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeAXPYThreeVectors"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
