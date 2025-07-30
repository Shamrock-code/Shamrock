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
 * @file ComputeSumOverV.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeComputeSumOverV : public shamrock::solvergraph::INode {

        u32 block_size;

        public:
        NodeComputeSumOverV(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldRefs<T> &spans_field;
            const shamrock::solvergraph::ScalarEdge<T> &total_volume;
            shamrock::solvergraph::ScalarEdge<T> &mean_val;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> spans_field,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> total_volume,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> mean_val) {
            __internal_set_ro_edges({sizes, spans_field, total_volume});
            __internal_set_rw_edges({mean_val});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<T>>(1),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<T>>(2),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<T>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeSumOverV"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
