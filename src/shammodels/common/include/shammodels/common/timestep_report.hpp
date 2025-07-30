// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file timestep_report.hpp
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
#include <string>

namespace shammodels {

    inline std::string report_perf_timestep(
        f64 rate, u64 nobj, f64 tcompute, f64 mpi_timer, f64 alloc_time, size_t max_mem) {

        std::string log_rank_rate = shambase::format(
            "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:>10s} |",
            shamcomm::world_rank(),
            rate,
            nobj,
            tcompute,
            100 * (mpi_timer / tcompute),
            100 * (alloc_time / tcompute),
            shambase::readable_sizeof(max_mem));

        std::string gathered = "";
        shamcomm::gather_str(log_rank_rate, gathered);

        u64 obj_total        = shamalgs::collective::allreduce_sum(nobj);
        f64 max_t            = shamalgs::collective::allreduce_max(tcompute);
        f64 sum_t            = shamalgs::collective::allreduce_sum(tcompute);
        f64 sum_mpi          = shamalgs::collective::allreduce_sum(mpi_timer);
        f64 sum_alloc        = shamalgs::collective::allreduce_sum(alloc_time);
        size_t sum_mem_total = shamalgs::collective::allreduce_sum(max_mem);

        std::string log_all_rate = shambase::format(
            "\n|  all |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:>10s} |",
            f64(obj_total) / max_t,
            obj_total,
            max_t,
            100 * (sum_mpi / sum_t),
            100 * (sum_alloc / sum_t),
            shambase::readable_sizeof(sum_mem_total));

        std::string print = "";

        // clang-format off
        if (shamcomm::world_rank() == 0) {
            print = "processing rate infos : \n";
            print += ("---------------------------------------------------------------------------------------\n");
            print += ("| rank |  rate  (N.s^-1)  |     Nobj    | t compute (s) |  MPI   | alloc |  mem (max) |\n");
            if (shamcomm::world_size() > 1) {
                print += ("------------------------------------- Per ranks ---------------------------------------");
                print += (gathered) + "\n";
              //print += ("| rank |  rate  (N.s^-1)  |     Nobj    | t compute (s) | interf | alloc | mem (max) |\n");
                print += ("---------<sum N>/<max t> ----- <sum> ------- <max> ------ <avg> -- <avg> --- <sum> ----");
                print += (log_all_rate) + "\n";
            }else{
                print += ("---------------------------------------------------------------------------------------");
                print += (gathered) + "\n";
            }
            print += ("---------------------------------------------------------------------------------------");
        }
        // clang-format on

        return print;
    }

} // namespace shammodels
