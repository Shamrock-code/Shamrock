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
 * @file DeviceBufferEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solver graph edge wrapping a global device buffer.
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class DeviceBufferEdge : public IEdgeNamed {

        public:
        sham::DeviceBuffer<T> value;

        DeviceBufferEdge(std::string name, std::string texsymbol)
            : IEdgeNamed(std::move(name), std::move(texsymbol)),
              value(0, shamsys::instance::get_compute_scheduler_ptr()) {}

        inline virtual void free_alloc() override { value.resize(0); }
    };

} // namespace shamrock::solvergraph
