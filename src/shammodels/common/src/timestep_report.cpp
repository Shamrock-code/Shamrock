// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file timestep_report.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/common/timestep_report.hpp"
#include <string>
#include <variant>

namespace shammodels {

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
} // namespace shammodels

std::string shammodels::report_perf_timestep(
    f64 rate,
    u64 nobj,
    u64 npatch,
    f64 tcompute,
    f64 mpi_timer,
    f64 alloc_time_device,
    f64 alloc_time_host,
    size_t max_mem_device,
    size_t max_mem_host) {

    std::vector<f64> rate_all_ranks              = shamalgs::collective::gather(rate);
    std::vector<u64> nobj_all_ranks              = shamalgs::collective::gather(nobj);
    std::vector<u64> npatch_all_ranks            = shamalgs::collective::gather(npatch);
    std::vector<f64> tcompute_all_ranks          = shamalgs::collective::gather(tcompute);
    std::vector<f64> mpi_timer_all_ranks         = shamalgs::collective::gather(mpi_timer);
    std::vector<f64> alloc_time_device_all_ranks = shamalgs::collective::gather(alloc_time_device);
    std::vector<f64> alloc_time_host_all_ranks   = shamalgs::collective::gather(alloc_time_host);
    std::vector<size_t> max_mem_device_all_ranks = shamalgs::collective::gather(max_mem_device);
    std::vector<size_t> max_mem_host_all_ranks   = shamalgs::collective::gather(max_mem_host);

    if (shamcomm::world_rank() != 0) {
        return "";
    }

    // be careful with overflows
    u64 obj_total    = std::accumulate(nobj_all_ranks.begin(), nobj_all_ranks.end(), 0_u64);
    u64 npatch_total = std::accumulate(npatch_all_ranks.begin(), npatch_all_ranks.end(), 0_u64);
    f64 max_t        = *std::max_element(tcompute_all_ranks.begin(), tcompute_all_ranks.end());
    f64 sum_t        = std::accumulate(tcompute_all_ranks.begin(), tcompute_all_ranks.end(), 0.0);
    f64 sum_mpi      = std::accumulate(mpi_timer_all_ranks.begin(), mpi_timer_all_ranks.end(), 0.0);
    f64 sum_alloc_device = std::accumulate(
        alloc_time_device_all_ranks.begin(), alloc_time_device_all_ranks.end(), 0.0);
    f64 sum_alloc_host
        = std::accumulate(alloc_time_host_all_ranks.begin(), alloc_time_host_all_ranks.end(), 0.0);
    size_t sum_mem_device_total
        = std::accumulate(max_mem_device_all_ranks.begin(), max_mem_device_all_ranks.end(), 0_u64);
    size_t sum_mem_host_total
        = std::accumulate(max_mem_host_all_ranks.begin(), max_mem_host_all_ranks.end(), 0_u64);

    static constexpr u32 cols_count = 9;

    using Table = table<cols_count>;

    Table table;

    table.add_double_rule();
    table.add_data(
        {"rank",
         "rate (N/s)",
         "Nobj",
         "Npatch",
         "tstep",
         "MPI",
         "alloc d% h%",
         "mem (max) d",
         "mem (max) h"},
        Table::center);
    table.add_double_rule();
    for (u32 i = 0; i < shamcomm::world_size(); i++) {
        table.add_data(
            {shambase::format("{:<4}", i),
             shambase::format("{:.4e}", rate_all_ranks[i]),
             shambase::format("{:}", nobj_all_ranks[i]),
             shambase::format("{:}", npatch_all_ranks[i]),
             shambase::format("{:.3e}", tcompute_all_ranks[i]),
             shambase::format("{:.1f}%", 100 * (mpi_timer_all_ranks[i] / tcompute_all_ranks[i])),
             shambase::format(
                 "{:>.1f}% {:<.1f}%",
                 100 * (alloc_time_device_all_ranks[i] / tcompute_all_ranks[i]),
                 100 * (alloc_time_host_all_ranks[i] / tcompute_all_ranks[i])),
             shambase::format("{}", shambase::readable_sizeof(max_mem_device_all_ranks[i])),
             shambase::format("{}", shambase::readable_sizeof(max_mem_host_all_ranks[i]))},
            Table::right);
    }
    if (shamcomm::world_size() > 1) {
        table.add_rulled_data(
            {"", "<sum N/max t>", "<sum>", "<sum>", "<max>", "<avg>", "<avg>", "<sum>", "<sum>"});
        table.add_data(
            {"all",
             shambase::format("{:.4e}", f64(obj_total) / max_t),
             shambase::format("{:}", obj_total),
             shambase::format("{:}", npatch_total),
             shambase::format("{:.3e}", max_t),
             shambase::format("{:.1f}%", 100 * (sum_mpi / sum_t)),
             shambase::format(
                 "{:>.1f}% {:<.1f}%",
                 100 * (sum_alloc_device / sum_t),
                 100 * (sum_alloc_host / sum_t)),
             shambase::format("{}", shambase::readable_sizeof(sum_mem_device_total)),
             shambase::format("{}", shambase::readable_sizeof(sum_mem_host_total))},
            Table::right);
    }
    table.add_rule();

    return "Timestep perf report:" + table.render();
}
