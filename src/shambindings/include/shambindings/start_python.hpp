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
 * @file start_python.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <string>

namespace shambindings {

    /**
     * @brief set the value of sys.path before init
     *
     * This function will throw if bindings were not initialized in embed mode
     */
    void setpypath(std::string path);

    /**
     * @brief set the value of sys.path before init from the supplied binary
     *
     * This function will throw if bindings were not initialized in embed mode
     */
    void setpypath_from_binary(std::string binary_path);

    /**
     * @brief Start shamrock embded ipython interpreter
     *
     * This function will throw if bindings were not initialized in embed mode
     *
     * @warning This function shall not be called if more than one processes are running
     * @param do_print print log at python startup
     */
    void start_ipython(bool do_print);

    /**
     * @brief run python runscript
     *
     * This function will throw if bindings were not initialized in embed mode
     *
     * @param do_print print log at python startup
     * @param file_path path to the runscript
     */
    void run_py_file(std::string file_path, bool do_print);

    /**
     * @brief Modify Python sys.path to point to one detected during cmake invocation
     *
     * This function will throw if bindings were not initialized in embed mode
     */
    void modify_py_sys_path(bool do_print);

} // namespace shambindings
