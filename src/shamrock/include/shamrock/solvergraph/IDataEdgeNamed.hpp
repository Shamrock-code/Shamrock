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
 * @file IDataEdgeNamed.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IDataEdge.hpp"

namespace shamrock::solvergraph {

    class IDataEdgeNamed : public IDataEdge {
        std::string name;
        std::string texsymbol;

        public:
        IDataEdgeNamed(std::string name, std::string texsymbol)
            : name(name), texsymbol(texsymbol) {}

        virtual std::string _impl_get_dot_label() const { return name; }
        virtual std::string _impl_get_tex_symbol() const { return "{" + texsymbol + "}"; }
    };

} // namespace shamrock::solvergraph
