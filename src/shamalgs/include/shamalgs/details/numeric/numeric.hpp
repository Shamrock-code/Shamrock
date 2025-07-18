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
     *   sham::DeviceBuffer<u64> d_counts = shamalgs::numeric::device_histogram<u64>(
     *       dev_sched, d_bin_edges, nbins, values, values.get_size());
     *   ```
     *
     *   bin_edges = {0.0, 1.0, 2.0, 3.0, 4.0} (4 bins: [0,1), [1,2), [2,3), [3,4))
     *   values = {0.5, 1.5, 2.5, 3.5, 2.1, 1.9, 0.1, 3.9}
     *   result = {2, 2, 2, 2}
     */
    template<class Tret, class T>
    sham::DeviceBuffer<Tret> device_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len);

    template<class T>
    inline sham::DeviceBuffer<u64> device_histogram_u64(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len) {
        return device_histogram<u64, T>(sched, bin_edges, nbins, values, len);
    }

    template<class T>
    inline sham::DeviceBuffer<u32> device_histogram_u32(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len) {
        return device_histogram<u32, T>(sched, bin_edges, nbins, values, len);
    }

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
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
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

    /**
     * @brief Structure holding the result of binning values for further computation.
     *
     * This struct contains the valid values that fall within the specified bins and the offsets
     * for each bin, allowing efficient per-bin computation.
     *
     * @tparam T The data type of the values.
     */
    template<class T>
    struct BinnedCompute {
        sham::DeviceBuffer<T>
            valid_values; ///< Values that are within the bin range, sorted by bin.
        sham::DeviceBuffer<u32> offsets_bins; ///< Offsets for each bin (size nbins+1).
    };

    /**
     * @brief Prepare binned data for per-bin computation.
     *
     * Filters and sorts the input values and keys into bins defined by bin_edges, returning the
     * valid values and the offsets for each bin. This is useful for custom per-bin reductions or
     * statistics.
     *
     * @tparam T The data type of the values and keys.
     * @param sched The device scheduler to run on.
     * @param bin_edges The edges of the bins (length == nbins + 1).
     * @param nbins The number of bins.
     * @param values The values to be binned (e.g., f(r)).
     * @param keys The keys used for binning (e.g., r).
     * @param len The number of elements in values/keys.
     * @return BinnedCompute<T> Structure containing valid values and bin offsets.
     *
     * Example:
     *   ```cpp
     *   auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
     *   sham::DeviceBuffer<double> bin_edges = ...;
     *   sham::DeviceBuffer<double> values = ...; // f(r)
     *   sham::DeviceBuffer<double> keys = ...;   // r
     *   u64 nbins = bin_edges.get_size() - 1;
     *   auto binned = shamalgs::numeric::binned_init_compute(dev_sched, bin_edges, nbins, values,
     * keys, values.get_size());
     *   // binned.valid_values, binned.offsets_bins
     *   ```
     */
    template<class T>
    BinnedCompute<T> binned_init_compute(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values, // ie f(r)
        const sham::DeviceBuffer<T> &keys,   // ie r
        u32 len);

    /**
     * @brief Perform a custom reduction or computation over values in each bin.
     *
     * This function applies a user-provided function to all values in each bin, allowing for
     * flexible per-bin reductions (e.g., sum, mean, min, max, etc.).
     *
     * @tparam T The data type of the values and keys.
     * @tparam Tret The return type of the per-bin computation.
     * @tparam Fct The type of the function to apply per bin. The function should have the
     * signature: Tret f(for_each_values, u32 bin_count) where for_each_values is a callable that
     * applies a function to each value in the bin.
     * @param sched The device scheduler to run on.
     * @param bin_edges The edges of the bins (length == nbins + 1).
     * @param nbins The number of bins.
     * @param values The values to be binned (e.g., f(r)).
     * @param keys The keys used for binning (e.g., r).
     * @param len The number of elements in values/keys.
     * @param fct The function to apply to each bin's values.
     * @return sham::DeviceBuffer<Tret> Buffer of computed values, one per bin.
     *
     * Example (per-bin sum):
     *   ```cpp
     *   auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
     *   sham::DeviceBuffer<double> bin_edges = ...;
     *   sham::DeviceBuffer<double> values = ...;
     *   sham::DeviceBuffer<double> keys = ...;
     *   u64 nbins = bin_edges.get_size() - 1;
     *   auto sums = shamalgs::numeric::binned_compute<double, double>(
     *       dev_sched, bin_edges, nbins, values, keys, values.get_size(),
     *       [](auto for_each_values, u32 bin_count) {
     *           double sum = 0;
     *           for_each_values([&](double v) { sum += v; });
     *           return sum;
     *       });
     *   ```
     */
    template<class T, class Tret, class Fct>
    sham::DeviceBuffer<Tret> binned_compute(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values, // ie f(r)
        const sham::DeviceBuffer<T> &keys,   // ie r
        u32 len,
        Fct &&fct) {

        auto bin_compute_in = binned_init_compute(sched, bin_edges, nbins, values, keys, len);
        auto &valid_values  = bin_compute_in.valid_values;
        auto &offsets_bins  = bin_compute_in.offsets_bins;

        auto &q = shambase::get_check_ref(sched).get_queue();

        sham::DeviceBuffer<Tret> bin_compute(nbins, sched);

        sham::kernel_call(
            q,
            sham::MultiRef{valid_values, offsets_bins},
            sham::MultiRef{bin_compute},
            nbins,
            [fct](
                u32 i,
                const T *__restrict valid_values,
                const u32 *__restrict offsets_bins,
                Tret *__restrict bin_averages) {
                u32 bin_start = offsets_bins[i];
                u32 bin_end   = offsets_bins[i + 1];
                u32 bin_count = bin_end - bin_start;

                auto for_each_values = [&](auto func) {
                    for (u32 j = bin_start; j < bin_end; j++) {
                        func(valid_values[j]);
                    }
                };

                bin_averages[i] = fct(for_each_values, bin_count);
            });

        return bin_compute;
    }

    /**
     * @brief Compute the sum of values in each bin.
     *
     * This function computes the sum of all values in each bin, using the keys to assign values to
     * bins.
     *
     * @tparam T The data type of the values and keys.
     * @param sched The device scheduler to run on.
     * @param bin_edges The edges of the bins (length == nbins + 1).
     * @param nbins The number of bins.
     * @param values The values to be summed (e.g., f(r)).
     * @param keys The keys used for binning (e.g., r).
     * @param len The number of elements in values/keys.
     * @return sham::DeviceBuffer<T> Buffer of sums, one per bin.
     *
     * Example:
     *   ```cpp
     *   auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
     *   sham::DeviceBuffer<double> bin_edges = ...;
     *   sham::DeviceBuffer<double> values = ...;
     *   sham::DeviceBuffer<double> keys = ...;
     *   u64 nbins = bin_edges.get_size() - 1;
     *   auto sums = shamalgs::numeric::binned_sum(dev_sched, bin_edges, nbins, values, keys,
     * values.get_size());
     *   ```
     */
    template<class T>
    sham::DeviceBuffer<T> binned_sum(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges, // r bins
        u64 nbins,
        const sham::DeviceBuffer<T> &values, // ie f(r)
        const sham::DeviceBuffer<T> &keys,   // ie r
        u32 len) {

        return binned_compute<T, T>(
            sched, bin_edges, nbins, values, keys, len, [](auto for_each_values, u32 bin_count) {
                T sum = 0;
                for_each_values([&](T val) {
                    sum += val;
                });
                return sum;
            });
    }

} // namespace shamalgs::numeric
