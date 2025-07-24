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
 * @file MicroBenchmark.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

namespace shamsys {

    /**
     * @brief Run latency & bandwidth benchmark
     * those benchmark where adapted from osu_microbenchmark
     * -     osu-micro-benchmarks/mpi/pt2pt/osu_bw.c
     * -     osu-micro-benchmarks/mpi/pt2pt/osu_latency.c
     */
    void run_micro_benchmark();

} // namespace shamsys
