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
 * @file SerialPatchTreeEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/exception.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <optional>
#include <stdexcept>

namespace shamrock::solvergraph {

    template<class Tvec>
    class SerialPatchTreeRefEdge : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        /**
         * @brief The patch tree.
         * @note this must be an optional because the edge needs to be freeable
         */
        std::optional<std::reference_wrapper<SerialPatchTree<Tvec>>> patch_tree;

        inline SerialPatchTree<Tvec> &get_patch_tree() const {
            if (!patch_tree.has_value()) {
                shambase::throw_with_loc<std::runtime_error>("Patch tree not set");
            }
            return patch_tree.value().get();
        }

        inline void free_alloc() { patch_tree = std::nullopt; };
    };

    template<class T>
    class SerialPatchTreeFieldRefEdge : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        /**
         * @brief The patch tree.
         * @note this must be an optional because the edge needs to be freeable
         */
        std::unique_ptr<shamrock::patch::PatchtreeField<T>> patch_tree_field;

        inline const shamrock::patch::PatchtreeField<T> &get_patch_tree_field() const {
            return shambase::get_check_ref(patch_tree_field);
        }

        inline void free_alloc() { patch_tree_field.reset(); };
    };

    template<class T>
    class SerialPatchFieldRefEdge : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        /**
         * @brief The patch tree.
         * @note this must be an optional because the edge needs to be freeable
         */
        std::unique_ptr<shamrock::patch::PatchField<T>> patch_tree_field;

        inline const shamrock::patch::PatchField<T> &get_patch_tree_field() const {
            return shambase::get_check_ref(patch_tree_field);
        }

        inline void free_alloc() { patch_tree_field.reset(); };
    };
} // namespace shamrock::solvergraph
