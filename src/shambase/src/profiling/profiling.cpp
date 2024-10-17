// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file chrome.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/profiling/chrome.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include <fstream>

void shambase::profiling::stack_entry_start(
    const std::source_location &fileloc,
    f64 t_start,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {}

void shambase::profiling::stack_entry_end(
    const std::source_location &fileloc,
    f64 t_start,
    f64 tend,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {}
