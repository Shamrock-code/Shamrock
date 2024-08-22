// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

template<class T, int ptrsep>
inline f64 get_bandwith() {

    constexpr u32 count   = 1e9;
    constexpr u32 buf_len = count + ptrsep;

    sycl::queue &q = shamsys::instance::get_compute_queue();

    T *ptr_read  = sycl::malloc_device<T>(buf_len, q);
    T *ptr_write = sycl::malloc_device<T>(buf_len, q);

    f64 duration_empty = shambase::timeitfor(
        [&]() {
            q.parallel_for(sycl::range<1>(count), [=](sycl::item<1> id) {
                 u64 gid    = id.get_linear_id();
                 u64 gidmul = (ptrsep) *gid % buf_len;

                 u32 write_addr = gid;
                 u32 read_addr  = gidmul;
             }).wait();
        },
        2);

    f64 duration = shambase::timeitfor(
        [&]() {
            q.parallel_for(sycl::range<1>(count), [=](sycl::item<1> id) {
                 u64 gid = id.get_linear_id();

                 u64 gidmul = (ptrsep) *gid % buf_len;

                 u32 write_addr = gid;
                 u32 read_addr  = gidmul;

                 ptr_write[write_addr] = ptr_read[read_addr];
             }).wait();
        },
        2);

    sycl::free(ptr_write, q);
    sycl::free(ptr_read, q);

    return 2 * f64(count) * sizeof(T) / (duration - duration_empty);
}

template<class T, int ptrsep>
void bench(std::vector<f64> &sep, std::vector<f64> &measured_bw) {
    sep.push_back(ptrsep * sizeof(T));
    measured_bw.push_back(get_bandwith<T, ptrsep>());

    shamcomm::logs::raw_ln(
        "sep =",
        ptrsep * sizeof(T),
        "   B = ",
        shambase::readable_sizeof(*(measured_bw.end() - 1)),
        "/s");
}

TestStart(Benchmark, "memory-pointer-div-perf", memorypointerdivperf, 1) {

    using T = double;

    std::vector<f64> sep;
    std::vector<f64> measured_bw;

    bench<T, 1>(sep, measured_bw);
    bench<T, 2>(sep, measured_bw);
    bench<T, 4>(sep, measured_bw);
    bench<T, 8>(sep, measured_bw);
    bench<T, 16>(sep, measured_bw);
    bench<T, 32>(sep, measured_bw);
    bench<T, 64>(sep, measured_bw);
    bench<T, 128>(sep, measured_bw);
    bench<T, 256>(sep, measured_bw);
    bench<T, 512>(sep, measured_bw);
    bench<T, 1024>(sep, measured_bw);
    bench<T, 2048>(sep, measured_bw);
    bench<T, 4096>(sep, measured_bw);
    bench<T, 8192>(sep, measured_bw);
}
