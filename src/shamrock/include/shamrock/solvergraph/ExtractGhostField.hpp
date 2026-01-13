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
 * @file ExtractGhostField.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief  This module allows to extract ghosts for a generic field such as density, velocity, etc
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class ExtractGhostField : public INode {

        public:
        ExtractGhostField() {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<T> &original_fields;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghosts;
            shamrock::solvergraph::PatchDataFieldDDShared<T> &ghost_fields;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<T>> original_fields,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghosts,
            std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<T>> ghost_fields) {
            __internal_set_ro_edges({original_fields, idx_in_ghosts});
            __internal_set_rw_edges({ghost_fields});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<T>>(0),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(1),
                get_rw_edge<shamrock::solvergraph::PatchDataFieldDDShared<T>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ExtractGhostField"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shamrock::solvergraph
