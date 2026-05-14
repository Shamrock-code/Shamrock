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
 * @file print.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Call a lambda at static or local scope construction time
 */

namespace shambase {

    /**
     * @brief Execute a lambda when a `call_lambda` object is constructed.
     *
     * Useful for static initialization side effects (registering data,
     * registering modules, etc.) without needing a named function.
     */
    struct call_lambda {

        /// Call the lambda on construction
        template<class Func>
        inline explicit call_lambda(Func &&f) {
            f();
        }
    };

} // namespace shambase
