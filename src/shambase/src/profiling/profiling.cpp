// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file profiling.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/profiling/chrome.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include <fstream>
#include <iostream>

#ifdef SHAMROCK_USE_NVTX
    #include <nvtx3/nvtx3.hpp>
#endif

std::string src_loc_to_name(const SourceLocation &loc) {
    return fmt::format(
        "{} ({}:{}:{})",
        loc.loc.function_name(),
        loc.loc.file_name(),
        loc.loc.line(),
        loc.loc.column());
}

bool use_complete_event = false;
f64 threshold           = 1e-5;

void shambase::profiling::stack_entry_start(
    const SourceLocation &fileloc,
    f64 t_start,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (!use_complete_event) {
        register_event_start(src_loc_to_name(fileloc), fileloc.loc.function_name(), t_start, 0, 0);
    }

    stack_entry_start_no_time(fileloc, name, category_name);
}

void shambase::profiling::stack_entry_end(
    const SourceLocation &fileloc,
    f64 t_start,
    f64 tend,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (use_complete_event) {
        if (tend - t_start > threshold) {
            register_event_complete(
                src_loc_to_name(fileloc), fileloc.loc.function_name(), t_start, tend, 0, 0);
        }
    } else {
        register_event_end(src_loc_to_name(fileloc), fileloc.loc.function_name(), tend, 0, 0);
    }

    stack_entry_end_no_time(fileloc, name, category_name);
}

void shambase::profiling::stack_entry_start_no_time(
    const SourceLocation &fileloc,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

#ifdef SHAMROCK_USE_NVTX
    // Push a NVTX range
    nvtxRangePush(fileloc.loc.function_name());
#endif
}

void shambase::profiling::stack_entry_end_no_time(
    const SourceLocation &fileloc,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

#ifdef SHAMROCK_USE_NVTX
    // Pop the NVTX range
    nvtxRangePop();
#endif
}
