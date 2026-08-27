// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambase/StlContainerConversion.hpp"
#include "shambase/memory.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
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
    bool is_all_true_sum_reduction(sham::DeviceBuffer<T> &buf, u32 cnt) {

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

    /// namespace to control implementation behavior
    namespace impl {

        /// Check all elements on host after copying the buffer back
        struct Host {
            static constexpr std::string_view variant_type_name = "host";
        };

        /// Check all elements via a sum reduction on device
        struct SumReduction {
            static constexpr std::string_view variant_type_name = "sum_reduction";
        };

        shamalgs::ImplVariantGlobal<Host, SumReduction> is_all_true_impl;

        /// Get list of available is_all_true implementations, as config json strings
        std::vector<std::string> get_default_impl_list_is_all_true() {
            return decltype(is_all_true_impl)::get_default_config_list();
        }

        /// Get the current implementation for is_all_true, as a config json string
        std::string get_current_impl_is_all_true() { return is_all_true_impl.get_current_config(); }

        /// Check if an implementation has been selected for is_all_true
        bool is_impl_set_is_all_true() { return is_all_true_impl.is_set(); }

        /// Set the implementation for is_all_true, from a config json string
        void set_impl_is_all_true(const std::string &impl) {
            shamlog_info_ln("algs", "setting is_all_true implementation to impl :", impl);
            is_all_true_impl.set(impl);
        }

        /// Select the default implementation for is_all_true
        void autoselect_impl_is_all_true() {
            is_all_true_impl.set(Host{});
            shamlog_info_ln(
                "algs",
                "defaulting is_all_true implementation to impl :",
                get_current_impl_is_all_true());
        }

    } // namespace impl

    template<class T>
    bool is_all_true(sham::DeviceBuffer<T> &buf, u32 cnt) {

        if (!impl::is_all_true_impl.is_set()) {
            impl::autoselect_impl_is_all_true();
        }

        return std::visit(
            shambase::overloaded{
                [&](impl::Host) {
                    return is_all_true_host(buf, cnt);
                },
                [&](impl::SumReduction) {
                    return is_all_true_sum_reduction(buf, cnt);
                },
            },
            impl::is_all_true_impl.get());
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
