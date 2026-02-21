// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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
#include "fmt/base.h"
#include <fstream>
#include <iostream>
#include <utility>

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

auto get_profiling = []() {
    const char *val = std::getenv("SHAM_PROFILING");
    if (val != nullptr) {
        if (std::string(val) == "1") {
            return true;
        } else if (std::string(val) == "0") {
            return false;
        }
    }
    return false;
};

auto get_nvtx = []() {
    const char *val = std::getenv("SHAM_PROF_USE_NVTX");
    if (val != nullptr) {
        if (std::string(val) == "1") {

            if (!get_profiling()) {
                fmt::println(
                    "-- SHAM_PROF_USE_NVTX is set to 1 but SHAM_PROFILING is not set to 1.\n"
                    "     please set SHAM_PROFILING=1 before SHAM_PROF_USE_NVTX=1");
            }

            return true;
        } else if (std::string(val) == "0") {
            return false;
        }
    }
    return false;
};

auto get_complete_event = []() {
    const char *val = std::getenv("SHAM_PROF_USE_COMPLETE_EVENT");
    if (val != nullptr) {
        if (std::string(val) == "1") {
            return true;
        } else if (std::string(val) == "0") {
            return false;
        }
    }
    return true;
};

auto get_threshold = []() {
    const char *val = std::getenv("SHAM_PROF_EVENT_RECORD_THRES");
    if (val != nullptr) {
        return std::stod(val);
    }
    return 1e-5;
};

bool enable_profiling   = get_profiling();
bool use_complete_event = get_complete_event();
f64 threshold           = get_threshold();
bool enable_nvtx        = get_nvtx();

bool shambase::profiling::is_profiling_enabled() { return enable_profiling; }

void shambase::profiling::set_enable_nvtx(bool enable) { enable_nvtx = enable; }
void shambase::profiling::set_enable_profiling(bool enable) { enable_profiling = enable; }
void shambase::profiling::set_use_complete_event(bool enable) { use_complete_event = enable; }
void shambase::profiling::set_event_record_threshold(f64 threshold_) { threshold = threshold_; }

void shambase::profiling::stack_entry_start(
    const SourceLocation &fileloc,
    f64 t_start,
    const std::optional<std::string> &name,
    const std::optional<std::string> &category_name) {

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
    const std::optional<std::string> &name,
    const std::optional<std::string> &category_name) {

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
    const std::optional<std::string> &name,
    const std::optional<std::string> &category_name) {

    if (enable_profiling && enable_nvtx) {
#ifdef SHAMROCK_USE_NVTX
        // Push a NVTX range
        nvtxRangePush(fileloc.loc.function_name());
#endif
    }
}

void shambase::profiling::stack_entry_end_no_time(
    const SourceLocation &fileloc,
    const std::optional<std::string> &name,
    const std::optional<std::string> &category_name) {

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
