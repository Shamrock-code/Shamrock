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
 * @file SGDirectPlummer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <sycl/sycl.hpp>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_G)                                      \
    X_RO(shamrock::solvergraph::FieldRefs<Tvec>, field_xyz)                                        \
    X_RW(shamrock::solvergraph::FieldRefs<Tvec>, field_axyz_ext)

namespace shammodels::sph::modules {

    template<class Tvec>
    class SGDirectPlummer : public shamrock::solvergraph::INode {

        using Tscal          = shambase::VecComponent<Tvec>;
        using PatchDataField = ::PatchDataField<Tvec>;

        Tscal epsilon;       ///< Gravitational softening length
        bool reference_mode; ///< If true this is a double loopm for debugging purposes

        public:
        explicit SGDirectPlummer(Tscal epsilon, bool reference_mode = false)
            : epsilon(epsilon), reference_mode(reference_mode) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline std::string _impl_get_label() const override { return "SGDirectPlummer"; }
        std::string _impl_get_tex() const override { return "TODO"; }

        protected:
        void _impl_evaluate_internal() override;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
