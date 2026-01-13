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
 * @file SignalCatch.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

namespace shamsys {
    namespace details {
        void signal_callback_handler(int signum);
    }
    void register_signals();
} // namespace shamsys
