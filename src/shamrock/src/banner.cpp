// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file banner.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "sham/term/tty.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/version.hpp"
#include <cstdlib>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>

// start allow utf-8
inline std::string shamrock_title_bar_big = "\n\
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████\n\
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░ \n\
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███   \n\
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████    \n\
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███   \n\
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███  \n\
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████\n\
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░ \n\
";
// end allow utf-8

// start allow utf-8
inline std::string very_rare_title_bar = "\n\
█▀▀▀▀▀█ ▀▄▄███▄▀  █▄   ▀  █▀▀▀▀▀█\n\
█ ███ █ ▄▄▀▄█ ▀▀██▀▀ █▄▀  █ ███ █\n\
█ ▀▀▀ █ ██▄ ▄  ▄▀█ ▀▄██▄▄ █ ▀▀▀ █\n\
▀▀▀▀▀▀▀ █ ▀▄▀ █ ▀▄█▄█▄▀ █ ▀▀▀▀▀▀▀\n\
▀▄  ▀▄▀█▀ █▄▀  █▄█▄ ▀███ █▀▀▀█▄ █\n\
▀ █ █▀▀█▀ ▄ ▄█▀▀▄ ▀▄▄ ▄▀▄█▄  ▀ ▀ \n\
█ ██▄▄▀▄▀ █▄▀▀▄ ▀▄▄▄▄ ▀▀█▄▀█ ▀▄ ▀\n\
▄▄█▄█▀▀▄  ██▀▄██▄██ ▀█ █▀██▄▀  ▀ \n\
██ ▀██▀▄██▄ ▀  ▄▄▀█▀█▀█▄█ ▀ ▀█▄█ \n\
▀▄█▀█ ▀█  █ ██ █▀█▄ ██▀▀██▄█ ▀ █ \n\
█ ▀▄ █▀▀ ▀ ▀▄▀▀ ▄▄▀█▀ ▄   █▀▀█▄▄ \n\
   ▄██▀█▄ ▄▀▀▄▀ ▀▄██▄█▄▀▀█▄ ▄▄▀█▄\n\
▀▀▀ ▀ ▀▀▄ ██ █████  ▀██▄█▀▀▀█▀▄▀ \n\
█▀▀▀▀▀█ ▀▀▀▄  ██▄ ▀▄  ▄ █ ▀ █▀  ▄\n\
█ ███ █ ▀▄▄▀██▀ ▀ ▄█▄ ▀▄▀▀████▄▄▀\n\
█ ▀▀▀ █  ▀ ▄█▄▀█▄▄█ ▀█ ▄▄▀█▀▀▀   \n\
▀▀▀▀▀▀▀ ▀ ▀      ▀ ▀▀▀▀▀▀▀▀  ▀  ▀\n\
";
// end allow utf-8

namespace {

    std::optional<std::string> get_banner_pic_path() {
        constexpr const char *kitty_banner_path
            = "../doc/mkdocs/docs/assets/no_background_nocolor.png";
        if (std::filesystem::exists(kitty_banner_path)) {
            return kitty_banner_path;
        }
        return std::nullopt;
    }

    bool try_print_picture() {
        if (!sham::term::is_a_tty()) {
            return false;
        }

        const char *term = std::getenv("TERM");
        if (term == nullptr || std::string(term) != "xterm-kitty") {
            return false;
        }

        if (std::optional<std::string> banner_path = get_banner_pic_path()) {
            std::system(("kitten icat " + *banner_path).c_str());
            return true;
        }

        return false;
    }

    void print_standard_title_bar() {
        // if (!try_print_picture()) { // need more testing on the kitty banner picture
        logger::raw_ln(shamrock_title_bar_big);
        // }
    }

} // namespace

namespace shamrock {

    std::string get_date_hour_string() {
        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    u64 pure_random_number() { return std::random_device{}(); }

    void print_title_bar() {
        if (pure_random_number() % 100 == 0) {
            logger::raw_ln(very_rare_title_bar);
        } else {
            print_standard_title_bar();
        }

        logger::raw_ln(
            shambase::term_colors::col8b_cyan()
            + "Copyright (c) 2021-2026 Timothée David--Cléris (tim.shamrock@proton.me)"
            + shambase::term_colors::reset());
        logger::raw_ln(
            shambase::term_colors::col8b_cyan() + "SPDX-License-Identifier"
            + shambase::term_colors::reset() + " : CeCILL Free Software License Agreement v2.1");
        logger::raw_ln(
            shambase::term_colors::col8b_cyan() + "Start time" + shambase::term_colors::reset()
            + " : " + get_date_hour_string());

        logger::print_faint_row();

        logger::raw_ln(
            "\n" + shambase::term_colors::col8b_cyan() + "Shamrock version "
            + shambase::term_colors::reset() + ": " + version_string + "\n");

        if (is_git) {
            logger::raw_ln(
                shambase::term_colors::col8b_cyan() + "Git infos " + shambase::term_colors::reset()
                + ":\n" + shambase::trunc_str(git_info_str, 512));
        }
    }

    void code_init_done_log() {

        // start allow utf-8
        auto lines = std::vector<std::string>{
            // Someone that coded too much here
            "Now it's time to " + shambase::term_colors::col8b_cyan()
                + shambase::term_colors::blink() + "ROCK" + shambase::term_colors::reset() + ".",
            "Shamrock rolls - no time for moss.", // Rolling stone gathers no moss.
            "Shamrock's live - go with the flow.",
            "Shamrock - as solid as a rock.",
            "Shamrock's stable and steady as a rock.",
            "Shamrock initialized - no cracks in this rock.",
            "Shamrock is ready to eat cheese (melted) and bread.",
            "Are you sure you want to work today?",
            "No holidays for the Shamrock ... (yeah, this was a PhD at some point)",
            "-[--->+<]>--.>+[----->+++<]>+.-------.++++++++++++.+++++.---.------------.++++++++.",
            "CPU hours to burn? We don't do such thing here.",
            "Are you burning GPUs or CPUs today?",
            R"=(
While you wait for this simulation to run, give that cat a hug!

  |\__/,|   (`\
_.|o o  |_   ) )
-(((---(((--------
          )=",

            // Someone that started on oumuamua
            "Shamrock your way to a brighter day!",
            "Node hours to burn? Leaf it to me.",
            "Ready for some shamazing simulations?",
            "SHAMROCKがきれいですね ~",
            "シャムロック",

            // by the coagulator
            "We're not here to make seagulls laugh",

            // in places
            "日本でも使ている", // used in japan
            "Pretty sure Aussies use that too, mate."};
        // end allow utf-8

        auto get_sentence = [&]() {
            f64 t   = shambase::details::get_wtime();
            u64 idx = static_cast<u64>(std::floor(
                          t * 2503'09713 // you wont guess what this stands for
                          ))
                      % lines.size();
            return lines[idx];
        };

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
            logger::raw_ln(
                " - Code init:",
                shambase::term_colors::col8b_green() + "DONE" + shambase::term_colors::reset()
                    + ".",
                get_sentence());
            logger::print_faint_row();
        }
    }

} // namespace shamrock
