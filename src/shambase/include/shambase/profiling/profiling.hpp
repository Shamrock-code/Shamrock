// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file profiling.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <optional>
#include <string>

namespace shambase::profiling {

    void set_enable_nvtx(bool enable_nvtx);
    void set_enable_profiling(bool enable_profiling);
    void set_use_complete_event(bool use_complete_event);
    void set_event_record_threshold(f64 threshold);

    void stack_entry_start(
        const SourceLocation &fileloc,
        f64 t_start,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    void stack_entry_end(
        const SourceLocation &fileloc,
        f64 t_start,
        f64 tend,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    void stack_entry_start_no_time(
        const SourceLocation &fileloc,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    void stack_entry_end_no_time(
        const SourceLocation &fileloc,
        std::optional<std::string> name          = std::nullopt,
        std::optional<std::string> category_name = std::nullopt);

    void register_counter_val(const std::string &name, f64 time, f64 val);

} // namespace shambase::profiling
