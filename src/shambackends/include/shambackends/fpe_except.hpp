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
 * @file fpe_except.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcomm/logs.hpp"
#include <fenv.h>

namespace sham {

    /**
     * @brief Enable floating point exceptions.
     *
     * This function enables all floating point exceptions using the fenv.h
     * header. This is useful for catching and handling floating point errors
     * that could lead to NaN or Inf values during computation.
     *
     * @note This function is only available on platforms that support the
     *       fenv.h header.
     *
     */
    inline void enable_fpe_exceptions() {
#ifdef __USE_GNU
        // we do not enable FE_INVALID as well as FE_UNDERFLOW
        // since they trigger exceptions in python lib ... like come on ...
        feenableexcept(FE_DIVBYZERO | FE_INVALID);
#else
        shamcomm::logs::warn_ln(
            "Backends", "Floating point exceptions are not supported on this platform.");
#endif
    }

} // namespace sham
