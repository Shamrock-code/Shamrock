// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file equals.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Element-wise equality comparison algorithms for buffers
 *
 * This header provides parallel algorithms to compare two buffers element-wise
 * for equality. The functions perform element-wise comparisons across entire buffers
 * or specified ranges, returning true only if all corresponding elements are equal.
 *
 * The algorithms come in several variants:
 * - `equals` for sycl::buffer: Direct buffer-based processing (deprecated)
 * - `equals` for sham::DeviceBuffer: USM-based processing
 * - `equals_ptr` and `equals_ptr_s`: Comparison of unique_ptr-wrapped buffers
 */

#include "shambase/exception.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Compare elements between two sycl::buffers for equality
     *
     * Performs element-wise comparison between two buffers to determine if all
     * corresponding elements are equal. The function compares the first `cnt`
     * elements of each buffer and returns true only if all pairs match.
     *
     * @tparam T Element type - must support equality comparison
     * @param q sycl::queue for device execution
     * @param buf1 First buffer to compare
     * @param buf2 Second buffer to compare
     * @param cnt Number of elements to compare from the beginning of each buffer
     * @return true if all compared elements are equal, false otherwise
     *
     * @throws std::invalid_argument if either buffer is smaller than cnt
     *
     * @deprecated Use equals(const sham::DeviceScheduler_ptr &, sham::DeviceBuffer<T> &,
     * sham::DeviceBuffer<T> &, u32 ) instead.
     *
     * @code{.cpp}
     * // Example: Compare arrays element-wise
     * std::vector<i32> data1 = {1, 2, 3, 4, 5};
     * std::vector<i32> data2 = {1, 2, 3, 4, 5};
     * sycl::buffer<i32> buffer1(data1);
     * sycl::buffer<i32> buffer2(data2);
     * sycl::queue q;
     *
     * // Compare first 5 elements
     * bool are_equal = equals(q, buffer1, buffer2, 5);
     * // Result: true (all elements match)
     *
     * // Example with different data
     * std::vector<i32> data3 = {1, 2, 0, 4, 5};
     * sycl::buffer<i32> buffer3(data3);
     * bool different = equals(q, buffer1, buffer3, 5);
     * // Result: false (third element differs)
     * @endcode
     */
    template<class T>
    [[deprecated(
        "Use equals(const sham::DeviceScheduler_ptr &, sham::DeviceBuffer<T> &, "
        "sham::DeviceBuffer<T> &, u32 ) instead.")]]
    bool equals(sycl::queue &q, sycl::buffer<T> &buf1, sycl::buffer<T> &buf2, u32 cnt) {

        if (buf1.size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 1 is larger than cnt");
        }

        if (buf2.size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 2 is larger than cnt");
        }

        sycl::buffer<u8> res(cnt);
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc1{buf1, cgh, sycl::read_only};
            sycl::accessor acc2{buf2, cgh, sycl::read_only};

            sycl::accessor out{res, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range{cnt}, [=](sycl::item<1> item) {
                out[item] = sham::equals(acc1[item], acc2[item]);
            });
        });

        return shamalgs::primitives::is_all_true(res, cnt);
    }

    /**
     * @brief Compare elements between two sham::DeviceBuffers for equality
     *
     * Performs element-wise comparison between two device buffers to determine if all
     * corresponding elements are equal. The function compares the first `cnt` elements
     * of each buffer and returns true only if all pairs match. This is the preferred
     * method for USM-based buffer comparisons.
     *
     * @tparam T Element type - must support equality comparison
     * @param dev_sched Device scheduler pointer for execution context
     * @param buf1 First device buffer to compare
     * @param buf2 Second device buffer to compare
     * @param cnt Number of elements to compare from the beginning of each buffer
     * @return true if all compared elements are equal, false otherwise
     *
     * @throws std::invalid_argument if either buffer is smaller than cnt
     *
     * @note Returns true immediately if both buffers are the same object
     * @note Uses parallel device execution for the comparison operation
     *
     * @code{.cpp}
     * // Example: Compare device buffers element-wise
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     * std::vector<f32> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
     * std::vector<f32> data2 = {1.0f, 2.0f, 3.0f, 4.0f};
     *
     * sham::DeviceBuffer<f32> buffer1(data1.size(), sched);
     * sham::DeviceBuffer<f32> buffer2(data2.size(), sched);
     *
     * // Initialize buffers with data
     * buffer1.copy_from_stdvec(data1);
     * buffer2.copy_from_stdvec(data2);
     *
     * // Compare first 4 elements
     * bool are_equal = equals(sched, buffer1, buffer2, 4);
     * // Result: true (all elements match)
     *
     * // Example with tolerance-based comparison
     * std::vector<f32> data3 = {1.0f, 2.0f, 3.1f, 4.0f};
     * sham::DeviceBuffer<f32> buffer3(data3.size(), sched);
     * buffer3.copy_from_stdvec(data3);
     * bool different = equals(sched, buffer1, buffer3, 4);
     * // Result: false (third element differs)
     * @endcode
     */
    template<class T>
    inline bool equals(
        const sham::DeviceScheduler_ptr &dev_sched,
        sham::DeviceBuffer<T> &buf1,
        sham::DeviceBuffer<T> &buf2,
        u32 cnt) {

        // kernel call does not support 0 elements
        if (cnt == 0) {
            return true;
        }

        // if the buffers are the same early return true
        if (&buf1 == &buf2) {
            return true;
        }

        if (buf1.get_size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 1 is larger than cnt");
        }

        if (buf2.get_size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 2 is larger than cnt");
        }

        sham::DeviceBuffer<u8> res(cnt, dev_sched);

        auto &q = shambase::get_check_ref(dev_sched).get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{buf1, buf2},
            sham::MultiRef{res},
            cnt,
            [](u32 i, const T *__restrict acc1, const T *__restrict acc2, u8 *__restrict out) {
                out[i] = sham::equals(acc1[i], acc2[i]);
            });

        return shamalgs::primitives::is_all_true(res, cnt);
    }

    /**
     * @brief Compare all elements between two sham::DeviceBuffers for equality
     *
     * Performs element-wise comparison between two device buffers to determine if all
     * corresponding elements are equal. This function automatically compares all elements
     * if both buffers have the same size, or returns false if sizes differ.
     *
     * @tparam T Element type - must support equality comparison
     * @param q Device scheduler pointer for execution context
     * @param buf1 First device buffer to compare
     * @param buf2 Second device buffer to compare
     * @return true if buffers have same size and all elements are equal, false otherwise
     *
     * @note Returns false immediately if buffer sizes differ
     * @note Convenience function that calls equals(q, buf1, buf2, buf1.get_size())
     *
     * @code{.cpp}
     * // Example: Compare entire device buffers
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     * std::vector<i32> keys = {10, 20, 30, 40};
     * std::vector<i32> values = {10, 20, 30, 40};
     *
     * sham::DeviceBuffer<i32> buffer1(keys.size(), sched);
     * sham::DeviceBuffer<i32> buffer2(values.size(), sched);
     *
     * buffer1.copy_from_stdvec(keys);
     * buffer2.copy_from_stdvec(values);
     *
     * // Compare entire buffers
     * bool are_equal = equals(sched, buffer1, buffer2);
     * // Result: true (all elements match)
     *
     * // Example with different sizes
     * std::vector<i32> shorter = {10, 20};
     * sham::DeviceBuffer<i32> buffer3(shorter.size(), sched);
     * buffer3.copy_from_stdvec(shorter);
     * bool size_diff = equals(sched, buffer1, buffer3);
     * // Result: false (different sizes)
     * @endcode
     */
    template<class T>
    inline bool equals(
        const sham::DeviceScheduler_ptr &q,
        sham::DeviceBuffer<T> &buf1,
        sham::DeviceBuffer<T> &buf2) {

        bool same_size = buf1.get_size() == buf2.get_size();
        if (!same_size) {
            return false;
        }

        return equals(q, buf1, buf2, buf1.get_size());
    }

    /**
     * @brief Compare all elements between two sycl::buffers for equality
     *
     * Performs element-wise comparison between two buffers to determine if all
     * corresponding elements are equal. This function automatically compares all elements
     * if both buffers have the same size, or returns false if sizes differ.
     *
     * @tparam T Element type - must support equality comparison
     * @param q sycl::queue for device execution
     * @param buf1 First buffer to compare
     * @param buf2 Second buffer to compare
     * @return true if buffers have same size and all elements are equal, false otherwise
     *
     * @note Returns false immediately if buffer sizes differ
     * @note Convenience function that calls equals(q, buf1, buf2, buf1.size())
     *
     * @deprecated Use equals with sham::DeviceBuffer instead
     *
     * @code{.cpp}
     * // Example: Compare entire sycl buffers
     * std::vector<f64> data1 = {1.0, 2.0, 3.0};
     * std::vector<f64> data2 = {1.0, 2.0, 3.0};
     * sycl::buffer<f64> buffer1(data1);
     * sycl::buffer<f64> buffer2(data2);
     * sycl::queue q;
     *
     * // Compare entire buffers
     * bool are_equal = equals(q, buffer1, buffer2);
     * // Result: true (all elements match)
     * @endcode
     */
    template<class T>
    bool equals(sycl::queue &q, sycl::buffer<T> &buf1, sycl::buffer<T> &buf2) {
        bool same_size = buf1.size() == buf2.size();
        if (!same_size) {
            return false;
        }

        return equals(q, buf1, buf2, buf1.size());
    }

    /**
     * @brief Compare elements between two unique_ptr-wrapped sycl::buffers with count
     *
     * Performs element-wise comparison between two unique_ptr-wrapped buffers to determine
     * if all corresponding elements are equal. The function handles null pointer cases
     * and compares the first `cnt` elements if both pointers are valid.
     *
     * @tparam T Element type - must support equality comparison
     * @param q sycl::queue for device execution
     * @param buf1 First unique_ptr-wrapped buffer to compare
     * @param buf2 Second unique_ptr-wrapped buffer to compare
     * @param cnt Number of elements to compare from the beginning of each buffer
     * @return true if both are null or all compared elements are equal, false otherwise
     *
     * @note Returns false if only one pointer is null
     * @note Returns true if both pointers are null
     * @note Uses equals(q, *buf1, *buf2, cnt) for actual comparison
     *
     * @deprecated Use sham::DeviceBuffer-based functions instead
     *
     * @code{.cpp}
     * // Example: Compare optional buffers with count
     * sycl::queue q;
     * std::vector<i32> data1 = {1, 2, 3, 4, 5};
     * std::vector<i32> data2 = {1, 2, 3, 4, 5};
     *
     * auto buf1 = std::make_unique<sycl::buffer<i32>>(data1);
     * auto buf2 = std::make_unique<sycl::buffer<i32>>(data2);
     *
     * // Compare first 3 elements
     * bool are_equal = equals_ptr_s(q, buf1, buf2, 3);
     * // Result: true (first 3 elements match)
     *
     * // Example with null pointers
     * std::unique_ptr<sycl::buffer<i32>> null_buf1, null_buf2;
     * bool both_null = equals_ptr_s(q, null_buf1, null_buf2, 0);
     * // Result: true (both are null)
     * @endcode
     */
    template<class T>
    bool equals_ptr_s(
        sycl::queue &q,
        const std::unique_ptr<sycl::buffer<T>> &buf1,
        const std::unique_ptr<sycl::buffer<T>> &buf2,
        u32 cnt) {
        bool same_alloc = bool(buf1) == bool(buf2);

        if (!same_alloc) {
            return false;
        }

        if (!bool(buf1)) {
            return true;
        }

        return equals(q, *buf1, *buf2, cnt);
    }

    /**
     * @brief Compare all elements between two unique_ptr-wrapped sycl::buffers
     *
     * Performs element-wise comparison between two unique_ptr-wrapped buffers to determine
     * if all corresponding elements are equal. The function handles null pointer cases
     * and compares all elements if both pointers are valid and buffer sizes match.
     *
     * @tparam T Element type - must support equality comparison
     * @param q sycl::queue for device execution
     * @param buf1 First unique_ptr-wrapped buffer to compare
     * @param buf2 Second unique_ptr-wrapped buffer to compare
     * @return true if both are null or all elements are equal, false otherwise
     *
     * @note Returns false if only one pointer is null
     * @note Returns true if both pointers are null
     * @note Uses equals(q, *buf1, *buf2) for actual comparison
     *
     * @deprecated Use sham::DeviceBuffer-based functions instead
     *
     * @code{.cpp}
     * // Example: Compare optional buffers entirely
     * sycl::queue q;
     * std::vector<f32> data1 = {1.0f, 2.0f, 3.0f};
     * std::vector<f32> data2 = {1.0f, 2.0f, 3.0f};
     *
     * auto buf1 = std::make_unique<sycl::buffer<f32>>(data1);
     * auto buf2 = std::make_unique<sycl::buffer<f32>>(data2);
     *
     * // Compare entire buffers
     * bool are_equal = equals_ptr(q, buf1, buf2);
     * // Result: true (all elements match)
     *
     * // Example with one null pointer
     * std::unique_ptr<sycl::buffer<f32>> null_buf;
     * bool one_null = equals_ptr(q, buf1, null_buf);
     * // Result: false (only one is null)
     * @endcode
     */
    template<class T>
    bool equals_ptr(
        sycl::queue &q,
        const std::unique_ptr<sycl::buffer<T>> &buf1,
        const std::unique_ptr<sycl::buffer<T>> &buf2) {
        bool same_alloc = bool(buf1) == bool(buf2);

        if (!same_alloc) {
            return false;
        }

        if (!bool(buf1)) {
            return true;
        }

        return equals(q, *buf1, *buf2);
    }
} // namespace shamalgs::primitives
