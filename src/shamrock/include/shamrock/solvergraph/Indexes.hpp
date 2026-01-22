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
 * @file Indexes.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include <memory>

namespace shamrock::solvergraph {

    template<class Tint>
    class Indexes : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;
        shambase::DistributedData<Tint> indexes;

        inline virtual void free_alloc() { indexes = {}; }

        static std::shared_ptr<Indexes<Tint>> make_shared(std::string name, std::string texsymbol) {
            return std::make_shared<Indexes<Tint>>(name, texsymbol);
        }
    };

} // namespace shamrock::solvergraph
