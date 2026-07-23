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
 * @file INullOptEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IEdge.hpp"

namespace shamrock::solvergraph {

    class INullOptEdge : public IEdge {
        public:
        INullOptEdge() : IEdge() {}

        virtual std::string _impl_get_dot_label() const { return "null_opt_edge"; }
        virtual std::string _impl_get_tex_symbol() const { return "null_opt_edge"; }
        virtual void free_alloc() {}
    };

    inline std::shared_ptr<INullOptEdge> make_null_opt_edge() {
        return std::make_shared<INullOptEdge>();
    }

    inline bool is_null_opt_edge(const std::shared_ptr<IEdge> &edge) {
        return std::dynamic_pointer_cast<INullOptEdge>(edge) != nullptr;
    }

} // namespace shamrock::solvergraph
