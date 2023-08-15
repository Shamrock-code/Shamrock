// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris
// <timothee.david--cleris@ens-lyon.fr> Licensed under CeCILL 2.1 License, see
// LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

namespace shamsys {
    
    /**
    * @brief Run latency & bandwidth benchmark
    * those benchmark where adapted from osu_microbenchmark
    * -     osu-micro-benchmarks/mpi/pt2pt/osu_bw.c
    * -     osu-micro-benchmarks/mpi/pt2pt/osu_latency.c
    */
    void run_micro_benchmark();

} // namespace shamsys