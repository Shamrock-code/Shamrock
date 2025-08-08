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
 * @file reduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction {

    template<class T>
    T sum(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    template<class T>
    T min(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    template<class T>
    T max(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
    shambase::VecComponent<T>
    dot_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    template<class T>
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

        return shamalgs::is_all_true(res, cnt);
    }

    template<class T>
    inline bool equals(
        const sham::DeviceScheduler_ptr &q,
        sham::DeviceBuffer<T> &buf1,
        sham::DeviceBuffer<T> &buf2,
        u32 cnt) {

        if (buf1.get_size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 1 is larger than cnt");
        }

        if (buf2.get_size() < cnt) {
            throw shambase::make_except_with_loc<std::invalid_argument>("buf 2 is larger than cnt");
        }

        bool is_same = (&buf1 == &buf2);

        sham::EventList depends_list;
        const T *acc1 = buf1.get_read_access(depends_list);
        const T *acc2 = (is_same) ? acc1 : buf2.get_read_access(depends_list);

        sycl::buffer<u8> res(cnt);
        auto e = q->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
            sycl::accessor out{res, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range{cnt}, [=](sycl::item<1> item) {
                out[item] = sham::equals(acc1[item], acc2[item]);
            });
        });

        buf1.complete_event_state(e);
        if (!is_same)
            buf2.complete_event_state(e);

        return shamalgs::is_all_true(res, cnt);
    }

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

    template<class T>
    bool has_nan(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

    template<class T>
    bool has_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

    template<class T>
    bool has_nan_or_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt);

    template<class T>
    bool equals(sycl::queue &q, sycl::buffer<T> &buf1, sycl::buffer<T> &buf2) {
        bool same_size = buf1.size() == buf2.size();
        if (!same_size) {
            return false;
        }

        return equals(q, buf1, buf2, buf1.size());
    }

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
} // namespace shamalgs::reduction
