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
 * @file ResidualDot.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    class ResidualDot : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<T>;

        public:
        ResidualDot() {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<T> &spans_phi_res;
            shamrock::solvergraph::ScalarEdge<Tscal> &res_ddot;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> res_ddot) {
            __internal_set_ro_edges({spans_phi_res});
            __internal_set_rw_edges({res_ddot});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<T>>(0),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ResidualDot"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
