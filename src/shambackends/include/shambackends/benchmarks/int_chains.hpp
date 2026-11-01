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
 * @file int_chains.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Integer ALU throughput benchmark (multiply chains vs add chains)
 *
 * Mirrors the structure of fma_chains.hpp, but on unsigned integers. Running the
 * multiply and the add variants gives the multiply/add throughput ratio, which
 * differs strongly across vendors: GCN-derived architectures issue a 32 bit
 * integer multiply at a fraction of the rate of an add, while architectures with
 * a dedicated integer pipe do not.
 */

#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <type_traits>

namespace sham::benchmarks {

    /// Which integer operation a chain is built from
    enum class IntChainOp {
        Mul, ///< multiply-add chains, 16 multiplies + 16 adds per rotation
        Add, ///< add-only chains, 32 adds per rotation
    };

// Both blocks issue the same number of ALU operations (32 per rotation), so the
// rates they produce are directly comparable.
#define IMAD_4(x, y)                                                                               \
    x = y * x + y;                                                                                 \
    y = x * y + x;                                                                                 \
    x = y * x + y;                                                                                 \
    y = x * y + x;
#define IMAD_16(x, y)                                                                              \
    IMAD_4(x, y);                                                                                  \
    IMAD_4(x, y);                                                                                  \
    IMAD_4(x, y);                                                                                  \
    IMAD_4(x, y);

#define IADD_4(x, y)                                                                               \
    x = y + x + y;                                                                                 \
    y = x + y + x;                                                                                 \
    x = y + x + y;                                                                                 \
    y = x + y + x;
#define IADD_16(x, y)                                                                              \
    IADD_4(x, y);                                                                                  \
    IADD_4(x, y);                                                                                  \
    IADD_4(x, y);                                                                                  \
    IADD_4(x, y);

    /**
     * @brief Kernel for the int_chains benchmark.
     *
     * Dependent integer chains, long enough to hide memory latency and expose the
     * integer issue rate. `T` must be unsigned so that the wraparound the chains
     * rely on is defined behaviour.
     *
     * @tparam T value type of the input and output vectors
     * @tparam op operation the chain is built from
     * @param i index of the element to process
     * @param nrotation number of chain rotations to apply
     * @param y0 initial value of the second chain register
     * @param in input vector
     * @param out output vector
     */
    template<class T, IntChainOp op>
    inline void int_chains(u32 i, int nrotation, T y0, T *__restrict in, T *__restrict out) {
        static_assert(std::is_unsigned_v<T>, "int_chains requires an unsigned type");

        T x = in[i];
        T y = y0;
        for (int j = 0; j < nrotation; j++) {
            if constexpr (op == IntChainOp::Mul) {
                IMAD_16(x, y);
            } else {
                IADD_16(x, y);
            }
        }
        out[i] = y;
    }

#undef IMAD_4
#undef IMAD_16
#undef IADD_4
#undef IADD_16

    /// Structure containing the results of an int_chains benchmark
    struct int_chains_result {
        std::string func_name; ///< Name of the function
        f64 seconds;           ///< Computation time in seconds
        f64 iops;              ///< Integer operations per second
        u32 nrotations;        ///< Number of rotations performed
    };

    /**
     * @brief Run the int_chains benchmark.
     *
     * @tparam T unsigned value type used in the benchmark
     * @tparam op operation the chains are built from
     * @param sched scheduler for the target device
     * @param N number of elements (independent chains) to process
     * @param time_threshold minimum wall-clock time to run the benchmark in seconds
     * @return benchmark results as an int_chains_result
     */
    template<class T, IntChainOp op>
    inline int_chains_result int_chains_bench(
        DeviceScheduler_ptr sched, int N, f64 time_threshold) {

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<T> x = {size_t(N), sched};
        sham::DeviceBuffer<T> y = {size_t(N), sched};

        const T x0 = T{3};
        const T y0 = T{5};

        x.fill(x0);
        y.fill(y0);

        sham::EventList depends_list;

        auto x_ptr = x.get_write_access(depends_list);
        auto y_ptr = y.get_write_access(depends_list);

        depends_list.wait();

        u32 nrotation = 8;
        double sec    = 0;

        auto run_bench = [&q, &N, &x_ptr, &y_ptr, y0](u32 nrotation) -> f64 {
            sham::EventList empty_list{};

            shambase::Timer t;
            t.start();
            auto e = q.submit(empty_list, [=](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
                    int_chains<T, op>(item.get_linear_id(), nrotation, y0, x_ptr, y_ptr);
                });
            });
            e.wait();
            t.stop();

            return t.elapsed_sec();
        };

        // warmup kernel
        run_bench(4);

        double ref = run_bench(0);

        for (;;) {

            sec = run_bench(nrotation);

            if (sec >= time_threshold || nrotation >= 256 * 256 * 4) {
                break;
            }

            nrotation *= 2;
        }

        x.complete_event_state(sycl::event{});
        y.complete_event_state(sycl::event{});

        sec -= ref;

        u64 op_per_thread = u64(nrotation) * 2_u64 * 16_u64;
        double op_count   = double(N) * double(op_per_thread);

        return {
            .func_name  = SourceLocation{}.loc.function_name(),
            .seconds    = sec,
            .iops       = op_count / sec,
            .nrotations = nrotation};
    }

} // namespace sham::benchmarks
