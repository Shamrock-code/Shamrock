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
 * @file CGHadamardProd.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeCGHadamardProd : public shamrock::solvergraph::INode {
        u32 block_size;

        public:
        NodeCGHadamardProd(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<T> &spans_phi_p;
            const shamrock::solvergraph::IFieldSpan<T> &spans_phi_Ap;
            shamrock::solvergraph::IFieldSpan<T> &spans_phi_hadamard_prod;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> spans_phi_p,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> spans_phi_Ap,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<T>> spans_phi_hadamard_prod) {
            __internal_set_ro_edges({sizes, spans_phi_p, spans_phi_Ap});
            __internal_set_rw_edges({spans_phi_hadamard_prod});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<T>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<T>>(0)};
        }

        void _impl_evaluate_internal();
        inline virtual std::string _impl_get_label() { return "CGHadamardProd"; };
        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
