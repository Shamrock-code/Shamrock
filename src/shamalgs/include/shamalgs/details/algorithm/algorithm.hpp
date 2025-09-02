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
 * @file algorithm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief main include file for the shamalgs algorithms
 *
 */

#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"

/**
 * @brief namespace to store algorithms implemented by shamalgs
 *
 */
namespace shamalgs::algorithm {

    /**
     * @brief Sort the buffer according to the key order
     *
     * @tparam T
     * @param q
     * @param buf_key
     * @param buf_values
     * @param len
     */
    template<class Tkey, class Tval>
    void sort_by_key(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {
        shamalgs::primitives::sort_by_key(q, buf_key, buf_values, len);
    }

    template<class Tkey, class Tval>
    void sort_by_key(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {
        shamalgs::primitives::sort_by_key(sched, buf_key, buf_values, len);
    }

    /**
     * @brief generate a buffer from a lambda expression based on the indexes
     *
     * @tparam Fct
     * @param q
     * @param len
     * @param func
     * @return sycl::buffer<typename std::invoke_result_t<Fct,u32>>
     */
    template<class Fct>
    inline sycl::buffer<typename std::invoke_result_t<Fct, u32>> gen_buffer_device(
        sycl::queue &q, u32 len, Fct &&func) {

        using ret_t = typename std::invoke_result_t<Fct, u32>;

        sycl::buffer<ret_t> ret(len);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor out{ret, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                out[item] = func(item.get_linear_id());
            });
        });

        return std::move(ret);
    }

    /**
     * @brief remap a buffer according to a given index map
     * result[i] = result[index_map[i]]
     *
     * This function can be used to apply a sort to another object
     *
     * @tparam T type of the buffer
     * @param q the sycl queue
     * @param buf the buffer to apply the remapping on
     * @param index_map the index map
     * @param len length of the index map
     */
    template<class T>
    sycl::buffer<T> index_remap(
        sycl::queue &q, sycl::buffer<T> &source_buf, sycl::buffer<u32> &index_map, u32 len);

    /**
     * @brief remap a buffer (with multiple variable per index) according to a given index map
     * result[i] = result[index_map[i]]
     *
     * This function can be used to apply a sort to another object
     *
     * @tparam T type of the buffer
     * @param q the sycl queue
     * @param buf the buffer to apply the remapping on
     * @param index_map the index map
     * @param len length of the index map
     * @param nvar the number of variable per index
     */
    template<class T>
    sycl::buffer<T> index_remap_nvar(
        sycl::queue &q,
        sycl::buffer<T> &source_buf,
        sycl::buffer<u32> &index_map,
        u32 len,
        u32 nvar);

    template<class T>
    void index_remap(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<T> &dest,
        sham::DeviceBuffer<u32> &index_map,
        u32 len);

    template<class T>
    void index_remap_nvar(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<T> &dest,
        sham::DeviceBuffer<u32> &index_map,
        u32 len,
        u32 nvar);

    template<class T>
    sham::DeviceBuffer<T> index_remap(
        const sham::DeviceScheduler_ptr &sched_ptr,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<u32> &index_map,
        u32 len) {

        sham::DeviceBuffer<T> dest(len, sched_ptr);
        index_remap<T>(sched_ptr, source, dest, index_map, len);
        return dest;
    }

    template<class T>
    sham::DeviceBuffer<T> index_remap_nvar(
        const sham::DeviceScheduler_ptr &sched_ptr,
        sham::DeviceBuffer<T> &source,
        sham::DeviceBuffer<u32> &index_map,
        u32 len,
        u32 nvar) {

        sham::DeviceBuffer<T> dest(len * nvar, sched_ptr);
        index_remap_nvar<T>(sched_ptr, source, dest, index_map, len, nvar);
        return dest;
    }

    /**
     * @brief generate a buffer such that for i in [0,len[, buf[i] = i
     *
     * @param q the queue to run on
     * @param len length of the buffer to generate
     * @return sycl::buffer<u32> the returned buffer
     */
    sycl::buffer<u32> gen_buffer_index(sycl::queue &q, u32 len);

    /**
     * @brief generate a buffer such that for i in [0,len[, buf[i] = i
     *
     * @param sched the scheduler to run on
     * @param len length of the buffer to generate
     * @return sham::DeviceBuffer<u32> the returned buffer
     */
    sham::DeviceBuffer<u32> gen_buffer_index_usm(sham::DeviceScheduler_ptr sched, u32 len);

    /**
     * @brief Fill a given buffer such that for i in [0,len[, buf[i] = i
     *
     * @param sched the scheduler to run on
     * @param len length of the buffer to fill
     * @param buf the buffer to fill
     */
    void fill_buffer_index_usm(
        sham::DeviceScheduler_ptr sched, u32 len, sham::DeviceBuffer<u32> &buf);

} // namespace shamalgs::algorithm
