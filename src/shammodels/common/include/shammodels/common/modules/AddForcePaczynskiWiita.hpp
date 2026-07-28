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
 * @file AddForcePaczynskiWiita.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Adds the acceleration from a Paczynski Wiita (1980) pseudo-newtonian potential.
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_G)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_c)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, central_mass)                                    \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_pos)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {

    template<class Tvec>
    class AddForcePaczynskiWiita : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForcePaczynskiWiita() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "AddForcePaczynskiWiita"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::common::modules

#undef NODE_EDGES
