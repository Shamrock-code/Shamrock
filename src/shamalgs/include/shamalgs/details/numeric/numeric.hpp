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
 * @file numeric.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl.hpp"

/**
 * @brief namespace containing the numeric algorithms of shamalgs
 *
 */
namespace shamalgs::numeric {

    /**
     * @brief Computes the exclusive sum of elements in a SYCL buffer.
     *
     * @tparam T The data type of elements in the buffer.
     * @param q The SYCL queue to use for computation.
     * @param buf1 The input buffer whose exclusive sum is to be computed.
     * @param len The number of elements in the buffer.
     * @return A new SYCL buffer containing the exclusive sum of the input buffer.
     */
    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    /**
     * @brief Compute the exclusive sum of a buffer on the device
     *
     * @param sched The scheduler to use for the computation
     * @param buf1 The buffer to sum
     * @param len The length of the sum
     * @return A new buffer which is the output of the sum
     */
    template<class T>
    sham::DeviceBuffer<T>
    exclusive_sum(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> inclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    template<class T>
    void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    /**
     * @brief Stream compaction algorithm
     *
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return std::tuple<sycl::buffer<u32>, u32> table of the index to extract and its size
     */
    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

    /**
     * @brief Stream compaction algorithm
     *
     * @param sched the device scheduler to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return sham::DeviceBuffer<u32> table of the index to extract
     */
    sham::DeviceBuffer<u32> stream_compact(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &buf_flags, u32 len);

    template<class T>
    struct histogram_result {
        sham::DeviceBuffer<u64> counts;
        sham::DeviceBuffer<T> bins_center;
        sham::DeviceBuffer<T> bins_width;
    };

    /**
     * @brief Compute the histogram of values between bin_edges.
     *
     * This function computes the histogram of the input values, counting how many values fall into
     * each bin defined by the bin_edges array. Only values within [bin_edges[0], bin_edges[nbins])
     * are counted; values outside this range are ignored.
     *
     * @tparam T The data type of the values and bin edges (e.g., float, double).
     * @param sched The device scheduler to run on.
     * @param bin_edges The edges of the bins (length == nbins + 1). Must be sorted in ascending
     * order.
     * @param nbins The number of bins (must be > 0, nbins = bin_edges.size() - 1).
     * @param values The values to compute the histogram on.
     * @param len The length of the values array.
     * @return sham::DeviceBuffer<u64> The counts in each bin (length == nbins).
     *
     * Example:
     *
     *   ```cpp
     *   auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
     *
     *   sham::DeviceBuffer<double> bin_edges = ...;
     *   u64 nbins = bin_edges.get_size() - 1;
     *
     *   sham::DeviceBuffer<double> values = ...;
     *
     *   sham::DeviceBuffer<u64> d_counts = shamalgs::numeric::device_histogram(
     *       dev_sched, d_bin_edges, nbins, values, values.get_size());
     *   ```
     *
     *   bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0} (4 bins: [0,1), [1,2), [2,3), [3,4))
     *   values = {0.5, 1.5, 2.5, 3.5, 2.1, 1.9, 0.1, 3.9}
     *   result = {2, 2, 2, 2}
     */
    template<class T>
    sham::DeviceBuffer<u64> device_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len);

    /**
     * @brief Compute the histogram and bin properties (center, width) for a set of values and bin
     * edges.
     *
     * This function returns the histogram counts, the center of each bin, and the width of each
     * bin.
     *
     * @tparam T The data type of the values and bin edges.
     * @param sched The device scheduler to run on.
     * @param bin_edges The edges of the bins (length == nbins + 1).
     * @param nbins The number of bins.
     * @param values The values to compute the histogram on.
     * @param len The length of the values array.
     * @return histogram_result<T> Structure containing counts, bin centers, and bin widths.
     */
    template<class T>
    histogram_result<T> device_histogram_full(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> bin_edges,
        u64 nbins,
        sham::DeviceBuffer<T> values,
        u32 len) {

        SHAM_ASSERT(nbins > 1); // at least a sup and a inf
        SHAM_ASSERT(bin_edges.get_size() == nbins + 1);

        auto &q = shambase::get_check_ref(sched).get_queue();

        sham::DeviceBuffer<u64> counts = device_histogram(sched, bin_edges, nbins, values, len);

        sham::DeviceBuffer<T> bins_center(nbins, sched);
        sham::DeviceBuffer<T> bins_width(nbins, sched);

        sham::kernel_call(
            q,
            sham::MultiRef{bin_edges},
            sham::MultiRef{bins_center, bins_width},
            nbins,
            [](u32 i,
               const T *__restrict bin_edges,
               T *__restrict bins_center,
               T *__restrict bins_width) {
                bins_center[i] = (bin_edges[i] + bin_edges[i + 1]) / 2;
                bins_width[i]  = bin_edges[i + 1] - bin_edges[i];
            });

        return {std::move(counts), std::move(bins_center), std::move(bins_width)};
    }

} // namespace shamalgs::numeric
