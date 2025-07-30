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
 * @file change_log_format.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

namespace shamsys {

    /**
     * @brief Change the log formatter according to the SHAMLOGFORMATTER and SHAMLOG_ERR_ON_EXCEPT
     * environment variables
     *
     * If SHAMLOGFORMATTER is 0, 1, 2, or 3, the log formatter will be changed to the corresponding
     * style. If SHAMLOG_ERR_ON_EXCEPT is 1, an exception handler callback will be set to generate
     * an error log when an exception is created.
     *
     * @note This function should be called before creating any logs.
     */
    void change_log_format();

} // namespace shamsys
