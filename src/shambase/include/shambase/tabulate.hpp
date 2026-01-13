// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file tabulate.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/string.hpp"
#include <string>
#include <variant>
#include <vector>

namespace shambase {

    template<size_t cols_count>
    struct table {
        struct rule {};
        struct double_rule {};
        struct rulled_data {
            std::array<std::string, cols_count> colnames;
        };
        enum positionning {
            left,
            right,
            center,
        };
        struct data {
            std::array<std::string, cols_count> cols;
            positionning position;
        };

        std::vector<std::variant<rule, double_rule, rulled_data, data>> table_lines;

        void add_rule() { table_lines.push_back(rule{}); }
        void add_double_rule() { table_lines.push_back(double_rule{}); }
        void add_rulled_data(std::array<std::string, cols_count> colnames) {
            table_lines.push_back(rulled_data{colnames});
        }
        void add_data(std::array<std::string, cols_count> cols, positionning position) {
            table_lines.push_back(data{cols, position});
        }

        std::array<size_t, cols_count> compute_widths() {
            std::array<size_t, cols_count> widths{};
            for (auto &line : table_lines) {
                if (data *data_line = std::get_if<data>(&line)) {
                    for (u32 i = 0; i < cols_count; i++) {
                        widths[i] = std::max(widths[i], data_line->cols[i].size());
                    }
                } else if (rulled_data *head_and_ruller_line = std::get_if<rulled_data>(&line)) {
                    for (u32 i = 0; i < cols_count; i++) {
                        widths[i] = std::max(widths[i], head_and_ruller_line->colnames[i].size());
                    }
                }
            }
            return widths;
        }

        std::string render() {

            std::array<size_t, cols_count> widths = compute_widths();

            std::string print = "";
            for (auto &line : table_lines) {
                if (data *data_line = std::get_if<data>(&line)) {
                    print += "\n|";
                    for (u32 i = 0; i < cols_count; i++) {
                        if (data_line->position == left) {
                            print += shambase::format(" {:<{}} |", data_line->cols[i], widths[i]);
                        } else if (data_line->position == right) {
                            print += shambase::format(" {:>{}} |", data_line->cols[i], widths[i]);
                        } else if (data_line->position == center) {
                            print += shambase::format(" {:^{}} |", data_line->cols[i], widths[i]);
                        }
                    }

                } else if (rulled_data *head_and_ruller_line = std::get_if<rulled_data>(&line)) {
                    std::string tmp = "+";
                    for (u32 i = 0; i < cols_count; i++) {
                        tmp += shambase::format(
                            " {:^{}} +", head_and_ruller_line->colnames[i], widths[i]);
                    }

                    auto neigh_char_is_good = [&](char c) -> bool {
                        return c == ' ' || c == '-' || c == '<' || c == '>' || c == '+';
                    };

                    std::vector<bool> set_to_space = std::vector<bool>(tmp.size(), false);
                    // if the next and previous chars are space and i am a space then set me to true
                    for (size_t i = 1; i < tmp.size() - 1; i++) {
                        if (tmp[i] == ' ' && neigh_char_is_good(tmp[i - 1])
                            && neigh_char_is_good(tmp[i + 1])) {
                            set_to_space[i] = true;
                        }
                    }
                    // replace the spaces by '-'
                    for (size_t i = 0; i < tmp.size() - 1; i++) {
                        if (set_to_space[i]) {
                            tmp[i] = '-';
                        }
                    }
                    print += "\n" + tmp;
                } else if (rule *rule_line = std::get_if<rule>(&line)) {
                    print += "\n+";
                    for (u32 i = 0; i < cols_count; i++) {
                        print += shambase::format(
                            "-{:<{}}-+", std::string(widths[i], '-'), widths[i]);
                    }
                } else if (double_rule *double_rule_line = std::get_if<double_rule>(&line)) {
                    print += "\n+";
                    for (u32 i = 0; i < cols_count; i++) {
                        print += shambase::format(
                            "={:<{}}=+", std::string(widths[i], '='), widths[i]);
                    }
                }
            }
            return print;
        }
    };

} // namespace shambase
