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
#include <memory>
#include <string>
#include <vector>

namespace shamrock::solvergraph {

    class INode;

    class IEdge : public shambase::WithUUID<IEdge, u64>, public IFreeable {
        public:
        inline std::string get_label() const { return _impl_get_dot_label(); }
        inline std::string get_tex_symbol() const { return _impl_get_tex_symbol(); }

        virtual std::string _impl_get_dot_label() const  = 0;
        virtual std::string _impl_get_tex_symbol() const = 0;

        /**
         * @brief Get the edges aggregated by this edge, if any.
         *
         * Container edges (see `EdgeList`) hold other edges, so a node reading a single container
         * slot really depends on all the contained edges. Returning them here lets the graph
         * tooling see through the container. Regular edges aggregate nothing and keep the default.
         */
        inline virtual std::vector<std::shared_ptr<IEdge>> get_sub_edges() const { return {}; }

        inline virtual ~IEdge() {}
    };

} // namespace shamrock::solvergraph
