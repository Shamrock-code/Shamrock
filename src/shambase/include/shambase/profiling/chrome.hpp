// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file chrome.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <string>

namespace shambase::profiling {

    void register_event_start(
        const std::string &name, const std::string &display_name, f64 t_start, u64 pid, u64 tid);

    void register_event_end(
        const std::string &name, const std::string &display_name, f64 tend, u64 pid, u64 tid);

    void register_metadata_thread_name(u64 pid, u64 tid, const std::string &name);

    void register_counter_val(u64 pid, f64 t, const std::string &name, f64 val);

} // namespace shambase::profiling
