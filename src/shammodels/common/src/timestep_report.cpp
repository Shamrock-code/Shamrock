// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file timestep_report.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/tabulate.hpp"
#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/common/timestep_report.hpp"
#include <numeric>
#include <string>
#include <variant>
#include <vector>

std::string shammodels::report_perf_timestep(
    f64 rate,
    u64 nobj,
    u64 npatch,
    f64 tcompute,
    f64 mpi_timer,
    f64 alloc_time_device,
    f64 alloc_time_host,
    size_t max_mem_device,
    size_t max_mem_host,
    shamsys::SystemMetrics system_metrics,
    bool report_power_usage) {

    __shamrock_stack_entry();

    std::vector<f64> rate_all_ranks              = shamalgs::collective::gather(rate);
    std::vector<u64> nobj_all_ranks              = shamalgs::collective::gather(nobj);
    std::vector<u64> npatch_all_ranks            = shamalgs::collective::gather(npatch);
    std::vector<f64> tcompute_all_ranks          = shamalgs::collective::gather(tcompute);
    std::vector<f64> mpi_timer_all_ranks         = shamalgs::collective::gather(mpi_timer);
    std::vector<f64> alloc_time_device_all_ranks = shamalgs::collective::gather(alloc_time_device);
    std::vector<f64> alloc_time_host_all_ranks   = shamalgs::collective::gather(alloc_time_host);
    std::vector<size_t> max_mem_device_all_ranks = shamalgs::collective::gather(max_mem_device);
    std::vector<size_t> max_mem_host_all_ranks   = shamalgs::collective::gather(max_mem_host);

    auto rank_metrics = (report_power_usage) ? shamsys::gather_rank_metrics(system_metrics)
                                             : std::vector<shamsys::SystemMetrics>{};

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

    shamsys::SystemMetrics aggregated_metrics = shamsys::aggregate_rank_metrics(rank_metrics);

    std::vector<shamsys::FormattedSystemMetrics> formatted_rank_metrics{};
    shamsys::FormattedSystemMetrics formatted_aggregated_metrics = {};
    if (report_power_usage) {
        for (const auto &metric : rank_metrics) {
            formatted_rank_metrics.push_back(shamsys::format_system_metrics(metric));
        }
        formatted_aggregated_metrics = shamsys::format_system_metrics(aggregated_metrics);
    }

    u32 cols_count = 9_u32;
    if (report_power_usage) {
        if (shamsys::support_rank_energy_consummed()) {
            cols_count += 1_u32;
        }
        if (shamsys::support_gpu_energy_consummed()) {
            cols_count += 1_u32;
        }
        if (shamsys::support_cpu_energy_consummed()) {
            cols_count += 1_u32;
        }
        if (shamsys::support_dram_energy_consummed()) {
            cols_count += 1_u32;
        }
    }

    using Table = shambase::table;

    Table table(cols_count);

    std::vector<std::string> header
        = {"rank",
           "rate (N/s)",
           "Nobj",
           "Npatch",
           "tstep",
           "MPI",
           "alloc d% h%",
           "mem (max) d",
           "mem (max) h"};
    if (report_power_usage) {
        if (shamsys::support_rank_energy_consummed()) {
            header.push_back("power");
        }
        if (shamsys::support_gpu_energy_consummed()) {
            header.push_back("gpu power");
        }
        if (shamsys::support_cpu_energy_consummed()) {
            header.push_back("cpu power");
        }
        if (shamsys::support_dram_energy_consummed()) {
            header.push_back("dram power");
        }
    }

    table.add_double_rule();
    table.add_data(header, Table::center);
    table.add_double_rule();
    for (u32 i = 0; i < shamcomm::world_size(); i++) {
        std::vector<std::string> row = {
            shambase::format("{:<4}", i),
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
            shambase::format("{}", shambase::readable_sizeof(max_mem_host_all_ranks[i])),
        };
        if (report_power_usage) {
            if (shamsys::support_rank_energy_consummed()) {
                row.push_back(formatted_rank_metrics[i].rank_power.value_or("N/A"));
            }
            if (shamsys::support_gpu_energy_consummed()) {
                row.push_back(formatted_rank_metrics[i].gpu_power.value_or("N/A"));
            }
            if (shamsys::support_cpu_energy_consummed()) {
                row.push_back(formatted_rank_metrics[i].cpu_power.value_or("N/A"));
            }
            if (shamsys::support_dram_energy_consummed()) {
                row.push_back(formatted_rank_metrics[i].dram_power.value_or("N/A"));
            }
        }
        table.add_data(row, Table::right);
    }
    if (shamcomm::world_size() > 1) {
        std::vector<std::string> ruled
            = {"", "<sum N/max t>", "<sum>", "<sum>", "<max>", "<avg>", "<avg>", "<sum>", "<sum>"};
        if (report_power_usage) {
            if (shamsys::support_rank_energy_consummed()) {
                ruled.push_back("<sum>");
            }
            if (shamsys::support_gpu_energy_consummed()) {
                ruled.push_back("<sum>");
            }
            if (shamsys::support_cpu_energy_consummed()) {
                ruled.push_back("<sum>");
            }
            if (shamsys::support_dram_energy_consummed()) {
                ruled.push_back("<sum>");
            }
        }
        table.add_rulled_data(ruled);
        std::vector<std::string> all_row = {
            "all",
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
            shambase::format("{}", shambase::readable_sizeof(sum_mem_host_total)),
        };
        if (report_power_usage) {
            if (shamsys::support_rank_energy_consummed()) {
                all_row.push_back(formatted_aggregated_metrics.rank_power.value_or("N/A"));
            }
            if (shamsys::support_gpu_energy_consummed()) {
                all_row.push_back(formatted_aggregated_metrics.gpu_power.value_or("N/A"));
            }
            if (shamsys::support_cpu_energy_consummed()) {
                all_row.push_back(formatted_aggregated_metrics.cpu_power.value_or("N/A"));
            }
            if (shamsys::support_dram_energy_consummed()) {
                all_row.push_back(formatted_aggregated_metrics.dram_power.value_or("N/A"));
            }
        }
        table.add_data(all_row, Table::right);
    }
    table.add_rule();

    return "Timestep perf report:" + table.render();
}
