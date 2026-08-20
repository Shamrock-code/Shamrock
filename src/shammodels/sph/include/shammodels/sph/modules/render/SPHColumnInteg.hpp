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
 * @file SPHColumnInteg.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief SPH column integration solver graph node.
 *
 */

#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shamrock/solvergraph/DeviceBufferEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, tree_reduction_level)                              \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, h_part)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, field_data)                                         \
    X_RO(shamrock::solvergraph::DeviceBufferEdge<shammath::Ray<Tvec>>, rays)                       \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::DeviceBufferEdge<T>, interpolated_field)

namespace shammodels::sph::modules {

    template<class Tvec, class T, template<class> class SPHKernel>
    class SPHColumnInteg : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal() override;

        inline std::string _impl_get_label() const override { return "SPHColumnInteg"; }

        std::string _impl_get_tex() const override;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
