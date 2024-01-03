// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file term_colors.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <string>

#define TERM_ESCAPTE_CHAR "\x1b["

namespace shambase {

    namespace details {

        struct TermColors {
            bool colors_on       = true;
            std::string esc_char = TERM_ESCAPTE_CHAR;

            std::string reset = TERM_ESCAPTE_CHAR "0m";

            std::string bold      = TERM_ESCAPTE_CHAR "1m";
            std::string faint     = TERM_ESCAPTE_CHAR "2m";
            std::string underline = TERM_ESCAPTE_CHAR "4m";
            std::string blink     = TERM_ESCAPTE_CHAR "5m";

            std::string col8b_black   = TERM_ESCAPTE_CHAR "30m";
            std::string col8b_red     = TERM_ESCAPTE_CHAR "31m";
            std::string col8b_green   = TERM_ESCAPTE_CHAR "32m";
            std::string col8b_yellow  = TERM_ESCAPTE_CHAR "33m";
            std::string col8b_blue    = TERM_ESCAPTE_CHAR "34m";
            std::string col8b_magenta = TERM_ESCAPTE_CHAR "35m";
            std::string col8b_cyan    = TERM_ESCAPTE_CHAR "36m";
            std::string col8b_white   = TERM_ESCAPTE_CHAR "37m";

            static TermColors get_config_nocolors() {
                return {false, "", "", "", "", "", "", "", "", "", "", "", "", "", ""};
            }

            static TermColors get_config_colors() { return {}; }
        };

        extern TermColors _int_term_colors;

    } // namespace details

    namespace term_colors {

        void enable_colors();

        void disable_colors();

        inline const std::string empty() { return ""; };
        inline const std::string reset() { return details::_int_term_colors.reset; };
        inline const std::string bold() { return details::_int_term_colors.bold; };
        inline const std::string faint() { return details::_int_term_colors.faint; };
        inline const std::string underline() { return details::_int_term_colors.underline; };
        inline const std::string blink() { return details::_int_term_colors.blink; };
        inline const std::string col8b_black() { return details::_int_term_colors.col8b_black; };
        inline const std::string col8b_red() { return details::_int_term_colors.col8b_red; };
        inline const std::string col8b_green() { return details::_int_term_colors.col8b_green; };
        inline const std::string col8b_yellow() { return details::_int_term_colors.col8b_yellow; };
        inline const std::string col8b_blue() { return details::_int_term_colors.col8b_blue; };
        inline const std::string col8b_magenta() { return details::_int_term_colors.col8b_magenta; };
        inline const std::string col8b_cyan() { return details::_int_term_colors.col8b_cyan; };
        inline const std::string col8b_white() { return details::_int_term_colors.col8b_white; };

    } // namespace term_colors

} // namespace shambase