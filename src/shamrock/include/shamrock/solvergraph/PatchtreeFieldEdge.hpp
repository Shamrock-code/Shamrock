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
 * @file PatchtreeFieldEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solvergraph edge holding a field defined on the nodes of the serial patch tree
 *
 */

#include "shambase/exception.hpp"
#include "shamrock/patch/PatchField.hpp"
#include "shamsolvergraph/edge/IEdgeNamed.hpp"
#include <stdexcept>

namespace shamrock::solvergraph {

    /**
     * @brief Edge holding a value per node of the serial patch tree (e.g. a max reduction of a
     * per patch value over the tree hierarchy)
     *
     * @tparam T type of the value stored on each tree node
     */
    template<class T>
    class PatchtreeFieldEdge : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        /// The field on the patch tree nodes
        shamrock::patch::PatchtreeField<T> patchtree_field;

        /// Get the underlying tree field buffer, throws if it is not allocated
        inline sycl::buffer<T> &get_buf() const {
            if (!bool(patchtree_field.internal_buf)) {
                shambase::throw_with_loc<std::runtime_error>("Patch tree field not set");
            }
            return *patchtree_field.internal_buf;
        }

        inline void free_alloc() { patchtree_field.reset(); };
    };

} // namespace shamrock::solvergraph
