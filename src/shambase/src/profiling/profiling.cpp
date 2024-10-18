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

// will not dump until changed
u32 pid                 = u32_max;
bool enable_nvtx        = false;
bool enable_profiling   = false;
bool use_complete_event = false;
f64 threshold           = 1e-5;

void shambase::profiling::set_pid(u32 pid_) { pid = pid_; }
void shambase::profiling::set_enable_nvtx(bool enable) { enable_nvtx = enable; }
void shambase::profiling::set_enable_profiling(bool enable) { enable_profiling = enable; }
void shambase::profiling::set_use_complete_event(bool enable) { use_complete_event = enable; }
void shambase::profiling::set_event_record_threshold(f64 threshold_) { threshold = threshold_; }

void shambase::profiling::stack_entry_start(
    const SourceLocation &fileloc,
    f64 t_start,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (enable_profiling) {

        if (!use_complete_event) {
            chrome::register_event_start(
                src_loc_to_name(fileloc), fileloc.loc.function_name(), t_start, 0, 0);
        }
    }
    stack_entry_start_no_time(fileloc, name, category_name);
}

void shambase::profiling::stack_entry_end(
    const SourceLocation &fileloc,
    f64 t_start,
    f64 tend,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (enable_profiling) {
        if (use_complete_event) {
            if (tend - t_start > threshold) {
                chrome::register_event_complete(
                    src_loc_to_name(fileloc), fileloc.loc.function_name(), t_start, tend, 0, 0);
            }
        } else {
            chrome::register_event_end(
                src_loc_to_name(fileloc), fileloc.loc.function_name(), tend, 0, 0);
        }
    }
    stack_entry_end_no_time(fileloc, name, category_name);
}

void shambase::profiling::stack_entry_start_no_time(
    const SourceLocation &fileloc,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (enable_profiling && enable_nvtx) {
#ifdef SHAMROCK_USE_NVTX
        // Push a NVTX range
        nvtxRangePush(fileloc.loc.function_name());
#endif
    }
}

void shambase::profiling::stack_entry_end_no_time(
    const SourceLocation &fileloc,
    std::optional<std::string> name,
    std::optional<std::string> category_name) {

    if (enable_profiling && enable_nvtx) {
#ifdef SHAMROCK_USE_NVTX
        // Pop the NVTX range
        nvtxRangePop();
#endif
    }
}

void shambase::profiling::register_counter_val(const std::string &name, f64 time, f64 val) {

    if (enable_profiling) {
        chrome::register_counter_val(0, time, name, val);
    }
}
