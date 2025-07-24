// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file IFreeable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

namespace shamrock::solvergraph {

    /**
     * @brief Interface for data edges that can free their allocated memory.
     *
     * Data edges should inherit from this interface if they manage memory
     * that needs to be freed at some point. This is useful for example when
     * using a memory pool to store data edge allocations.
     */
    class IFreeable {
        public:
        /// Free allocated memory.
        virtual void free_alloc() = 0;

        /// Virtual destructor
        virtual ~IFreeable() {}
    };

} // namespace shamrock::solvergraph
