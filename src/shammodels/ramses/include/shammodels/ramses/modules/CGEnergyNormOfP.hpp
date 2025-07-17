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
 * @file CGEnergyNormOfP.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeCGEnergyNormOfP : public shamrock::solvergraph::INode {

        u32 block_size;

        public:
        NodeCGEnergyNormOfP(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldRefs<T> &spans_phi_hadamard_prod;
            shamrock::solvergraph::ScalarEdge<T> &e_norm;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> spans_phi_hadamard_prod,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> e_norm) {
            __internal_set_ro_edges({sizes, spans_phi_hadamard_prod});
            __internal_set_rw_edges({e_norm});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<T>>(1),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<T>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "CGEnergyNormOfP"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
