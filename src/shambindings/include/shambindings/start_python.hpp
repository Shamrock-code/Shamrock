// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file start_python.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <string>

namespace shambindings {

#if defined(DOXYGEN) || defined(SHAMROCK_EXECUTABLE_BUILD)
    /**
     * @brief Start shamrock embded ipython interpreter
     *
     * This function is available only if the flag SHAMROCK_EXECUTABLE_BUILD is set
     *
     * @warning This function shall not be called if more than one processes are running
     * @param do_print print log at python startup
     */
    void start_ipython(bool do_print);

    /**
     * @brief run python runscript
     *
     * This function is available only if the flag SHAMROCK_EXECUTABLE_BUILD is set
     *
     * @param do_print print log at python startup
     * @param file_path path to the runscript
     */
    void run_py_file(std::string file_path, bool do_print);
#endif

} // namespace shambindings
