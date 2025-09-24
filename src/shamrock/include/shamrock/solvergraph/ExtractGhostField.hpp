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
 * @file ExtractGhostField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include <memory>
#include <string>

namespace shamrock::solvergraph {

    template<class T>
    class ExtractGhostField : public INode {

        public:
        ExtractGhostField() {}

        struct Edges {
            const shamrock::solvergraph::Field<T> &original_fields;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghots;
            shamrock::solvergraph::PatchDataFieldDDShared<T> &ghost_fields;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Field<T>> original_fields,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghosts,
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> ghost_fields) {
            __internal_set_ro_edges({original_fields, idx_in_ghosts});
            __internal_set_rw_edges({ghost_fields});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Field<T>>(0),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(1),
                get_rw_edge<shamrock::solvergraph::PatchDataFieldDDShared<T>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ExtractGhostField"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shamrock::solvergraph
