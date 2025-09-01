// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file is_all_true.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements functions to check if all elements in a buffer are non-zero (true).
 */

#include "shamalgs/primitives/is_all_true.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/kernel_call.hpp"

namespace {

    template<class T>
    bool is_all_true_host(sham::DeviceBuffer<T> &buf, u32 cnt) {

        {
            auto tmp = buf.copy_to_stdvec();

            for (u32 i = 0; i < cnt; i++) {
                if (tmp[i] == 0) {
                    return false;
                }
            }
        }

        return true;
    }

    template<class T>
    bool is_all_true_device(sham::DeviceBuffer<T> &buf, u32 cnt) {

        if (cnt == 0) {
            return true;
        }

        auto dev_sched = buf.get_dev_scheduler_ptr();

        sham::DeviceBuffer<u32> tmp(cnt, dev_sched);

        sham::kernel_call(
            shambase::get_check_ref(dev_sched).get_queue(),
            sham::MultiRef{buf},
            sham::MultiRef{tmp},
            cnt,
            [](u32 i, const T *in, u32 *out) {
                out[i] = in[i] != 0;
            });

        auto count_true = shamalgs::primitives::sum(dev_sched, tmp, 0, cnt);

        return count_true == cnt;
    }

} // namespace

namespace shamalgs::primitives {

    enum class IS_ALL_TRUE_IMPL { HOST, SUM_REDUCTION };
    IS_ALL_TRUE_IMPL is_all_true_impl = IS_ALL_TRUE_IMPL::HOST;

    std::vector<std::string> impl::get_impl_list_is_all_true() { return {"host", "sum_reduction"}; }

    void impl::set_impl_is_all_true(const std::string &impl, const std::string &param) {
        if (impl == "host") {
            is_all_true_impl = IS_ALL_TRUE_IMPL::HOST;
        } else if (impl == "sum_reduction") {
            is_all_true_impl = IS_ALL_TRUE_IMPL::SUM_REDUCTION;
        } else {
            throw std::invalid_argument("invalid implementation");
        }
    }

    template<class T>
    bool is_all_true(sham::DeviceBuffer<T> &buf, u32 cnt) {
        switch (is_all_true_impl) {
        case IS_ALL_TRUE_IMPL::HOST         : return is_all_true_host(buf, cnt);
        case IS_ALL_TRUE_IMPL::SUM_REDUCTION: return is_all_true_device(buf, cnt);
        default                             : throw std::invalid_argument("invalid implementation");
        }
    }

    template bool is_all_true(sham::DeviceBuffer<u8> &buf, u32 cnt);

} // namespace shamalgs::primitives

template<class T>
bool shamalgs::primitives::is_all_true(sycl::buffer<T> &buf, u32 cnt) {

    // TODO do it on GPU pleeeaze
    {
        sycl::host_accessor acc{buf, sycl::read_only};

        for (u32 i = 0; i < cnt; i++) {
            if (acc[i] == 0) {
                return false;
            }
        }
    }

    return true;
}

template bool shamalgs::primitives::is_all_true(sycl::buffer<u8> &buf, u32 cnt);
