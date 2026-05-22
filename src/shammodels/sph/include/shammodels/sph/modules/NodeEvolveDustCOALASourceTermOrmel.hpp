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
 * @file NodeEvolveDustCOALASourceTermOrmel.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <experimental/mdspan>
#include <vector>

#define NODE_EVOLVE_DUST_COALA_SOURCE_TERM_EDGES(X_RO, X_RW)                                       \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, rhodust_eps)                                    \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, massgrid)                          \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, tensor_tabflux_coag)               \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, alpha_turb)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mu)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mh)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, kb)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gpart_mass)                                          \
                                                                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                        \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, pressure)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, cs)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, ts)                                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, a_ext)                             \
                                            \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* to get rho_dust_j */                                                                        \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
                                                                                                   \
                                                                                                   \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, S_coag)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class NodeEvolveDustCOALASourceTermOrmel : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 nbins;

        public:
        NodeEvolveDustCOALASourceTermOrmel(u32 nbins) : nbins(nbins) {}

        EXPAND_NODE_EDGES(NODE_EVOLVE_DUST_COALA_SOURCE_TERM_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "NodeEvolveDustCOALASourceTermOrmel";
        };

        inline virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules

#undef NODE_EVOLVE_DUST_COALA_SOURCE_TERM_EDGES