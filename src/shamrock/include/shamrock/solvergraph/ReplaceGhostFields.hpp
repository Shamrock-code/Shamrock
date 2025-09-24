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
 * @file ReplaceGhostFields.hpp
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
    class ReplaceGhostFields : public INode {

        ReplaceGhostFields() {}

        struct Edges {
            const shamrock::solvergraph::PatchDataFieldDDShared<T> &ghost_fields;
            shamrock::solvergraph::Field<T> &fields;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> ghost_fields,
            std::shared_ptr<shamrock::solvergraph::Field<T>> fields) {
            __internal_set_ro_edges({ghost_fields});
            __internal_set_rw_edges({fields});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::PatchDataFieldDDShared<T>>(0),
                get_rw_edge<shamrock::solvergraph::Field<T>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_lable() { return "ReplaceGhostField"; };

        virtual std::string _impl_get_tex() { return "TODO"; };
    };
} // namespace shamrock::solvergraph
