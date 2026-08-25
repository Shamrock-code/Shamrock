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
 * @file IEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"
#include "shamsolvergraph/IFreeable.hpp"
#include <string>

namespace shamrock::solvergraph {

    class INode;

    class IEdge : public shambase::WithUUID<IEdge, u64>, public IFreeable {
        public:
        IEdge() = default;

        IEdge(const IEdge &)            = delete; /// would duplicate the uuid
        IEdge &operator=(const IEdge &) = delete; /// would duplicate the uuid

        /// Declared explicitly: the destructor below would otherwise suppress implicit move
        /// generation, and copy is deleted, leaving IEdge neither movable nor copyable.
        IEdge(IEdge &&) noexcept            = default;
        IEdge &operator=(IEdge &&) noexcept = default;

        inline std::string get_label() const { return _impl_get_dot_label(); }
        inline std::string get_tex_symbol() const { return _impl_get_tex_symbol(); }

        virtual std::string _impl_get_dot_label() const  = 0;
        virtual std::string _impl_get_tex_symbol() const = 0;

        inline virtual ~IEdge() {}
    };

} // namespace shamrock::solvergraph
