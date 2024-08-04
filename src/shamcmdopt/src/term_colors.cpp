// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file term_colors.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcmdopt/term_colors.hpp"

namespace shambase::details {
    TermColors _int_term_colors = TermColors::get_config_colors();
}

void shambase::term_colors::enable_colors() {
    details::_int_term_colors = details::TermColors::get_config_colors();
}

void shambase::term_colors::disable_colors() {
    details::_int_term_colors = details::TermColors::get_config_nocolors();
}
