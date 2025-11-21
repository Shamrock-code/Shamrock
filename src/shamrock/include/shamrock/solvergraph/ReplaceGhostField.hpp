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
 * @file ReplaceGhostField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include <memory>

namespace shamrock::solvergraph {

    template<class T>
    class ReplaceGhostField : public INode {

        public:
        ReplaceGhostField() {}

        struct Edges {
            shamrock::solvergraph::PatchDataFieldDDShared<T> &ghost_fields;
            shamrock::solvergraph::IFieldRefs<T> &fields;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> ghost_fields,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> fields) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({ghost_fields, fields});
        }

        inline Edges get_edges() {
            return Edges{
                get_rw_edge<shamrock::solvergraph::PatchDataFieldDDShared<T>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<T>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ReplaceGhostField"; };

        virtual std::string _impl_get_tex() { return "ReplaceGhostField"; };
    };
} // namespace shamrock::solvergraph
