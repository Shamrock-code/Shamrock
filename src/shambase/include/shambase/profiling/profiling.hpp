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
#include "shambase/cxxbackports/source_location.hpp"
#include <optional>
#include <string>

namespace shambase::profiling {



    void stack_entry_start(
        const std::source_location &fileloc, f64 t_start, std::optional<std::string> name = std::nullopt, std::optional<std::string> category_name = std::nullopt);

    void stack_entry_end(
        const std::source_location &fileloc,
        f64 t_start,
        f64 tend, std::optional<std::string> name = std::nullopt, std::optional<std::string> category_name = std::nullopt);

} // namespace shambase::profiling
