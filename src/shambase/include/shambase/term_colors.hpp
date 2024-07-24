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

/**
 * @brief Escape character for terminal control sequences.
 *
 * This is a control character that is used to initiate a terminal control
 * sequence. It is typically represented by the escape character `\x1b[`.
 */
#define TERM_ESCAPTE_CHAR "\x1b["

namespace shambase {

    namespace details {

        /**
         * @brief Struct representing the terminal colors to be used for printing.
         *
         * This struct contains the escape sequences for the different terminal
         * colors and can be used to print colored text.
         */
        struct TermColors {
            bool colors_on = true; ///< are colors on in this config

            /**
             * @brief Escape character for terminal control sequences.
             */
            std::string esc_char = TERM_ESCAPTE_CHAR;

            /**
             * @brief Escape sequence to reset the terminal text formatting.
             */
            std::string reset = TERM_ESCAPTE_CHAR "0m";

            /**
             * @brief Escape sequence to set bold text formatting.
             */
            std::string bold      = TERM_ESCAPTE_CHAR "1m";

            /**
             * @brief Escape sequence to set faint (dim) text formatting.
             */
            std::string faint     = TERM_ESCAPTE_CHAR "2m";

            /**
             * @brief Escape sequence to set underlined text formatting.
             */
            std::string underline = TERM_ESCAPTE_CHAR "4m";

            /**
             * @brief Escape sequence to set blinking text formatting.
             */
            std::string blink     = TERM_ESCAPTE_CHAR "5m";

            /**
             * @brief Escape sequence to set black text color.
             */
            std::string col8b_black   = TERM_ESCAPTE_CHAR "30m";

            /**
             * @brief Escape sequence to set red text color.
             */
            std::string col8b_red     = TERM_ESCAPTE_CHAR "31m";

            /**
             * @brief Escape sequence to set green text color.
             */
            std::string col8b_green   = TERM_ESCAPTE_CHAR "32m";

            /**
             * @brief Escape sequence to set yellow text color.
             */
            std::string col8b_yellow  = TERM_ESCAPTE_CHAR "33m";

            /**
             * @brief Escape sequence to set blue text color.
             */
            std::string col8b_blue    = TERM_ESCAPTE_CHAR "34m";

            /**
             * @brief Escape sequence to set magenta (pink) text color.
             */
            std::string col8b_magenta = TERM_ESCAPTE_CHAR "35m";

            /**
             * @brief Escape sequence to set cyan text color.
             */
            std::string col8b_cyan    = TERM_ESCAPTE_CHAR "36m";

            /**
             * @brief Escape sequence to set white text color.
             */
            std::string col8b_white   = TERM_ESCAPTE_CHAR "37m";

            /**
             * @brief Returns a `TermColors` struct with all colors disabled.
             */
            static TermColors get_config_nocolors() {
                return {false, "", "", "", "", "", "", "", "", "", "", "", "", "", ""};
            }

            /**
             * @brief Returns a `TermColors` struct with all colors enabled.
             */
            static TermColors get_config_colors() { return {}; }
        };

        /**
         * @brief Global instance of `TermColors`.
         *
         * This instance is used to store the escape sequences for the terminal colors.
         * The escape sequences are stored in the `TermColors` structure.
         *
         * @note This global variable is initialized in the implementation file.
         */
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
        inline const std::string col8b_magenta() {
            return details::_int_term_colors.col8b_magenta;
        };
        inline const std::string col8b_cyan() { return details::_int_term_colors.col8b_cyan; };
        inline const std::string col8b_white() { return details::_int_term_colors.col8b_white; };

    } // namespace term_colors

} // namespace shambase
