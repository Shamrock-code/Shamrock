// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file term_colors.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/term_colors.hpp"

namespace shambase::details {
    TermColors _int_term_colors = TermColors::get_config_colors();
}

void shambase::term_colors::enable_colors() {
    details::_int_term_colors = details::TermColors::get_config_colors();
}

void shambase::term_colors::disable_colors() {
    details::_int_term_colors = details::TermColors::get_config_nocolors();
}
