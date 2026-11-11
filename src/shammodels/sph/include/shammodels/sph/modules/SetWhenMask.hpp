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
 * @file SetWhenMask.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Solver graph node overwriting a field with a constant value where a mask is set.
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <string>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<u32>, mask)                                             \
    X_RW(shamrock::solvergraph::IFieldSpan<T>, field_to_set)

namespace shammodels::sph::modules {

    /**
     * @brief Overwrite every variable of a field with a constant where the per-object mask is
     * non-zero
     *
     * The mask holds one value per object, the field holds `nvar` values per object. All the
     * `nvar` values of an object with a non-zero mask are set to `val_to_set`, the others are
     * left untouched.
     *
     * @tparam T type of the field values
     */
    template<class T>
    class SetWhenMask : public shamrock::solvergraph::INode {

        T val_to_set;
        u32 nvar;

        public:
        SetWhenMask(T val_to_set, u32 nvar = 1) : val_to_set(val_to_set), nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SetWhenMask"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
