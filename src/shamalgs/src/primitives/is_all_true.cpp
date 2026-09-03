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
#include "shambackends/group_op.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/make_ndrange.hpp"

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

    template<class T>
    bool is_all_true_early_group_exit(sham::DeviceBuffer<T> &buf, u32 cnt, u32 group_size) {

        if (cnt == 0) {
            return true;
        }

        auto dev_sched = buf.get_dev_scheduler_ptr();
        auto &q        = dev_sched->get_queue();

        sham::DeviceBuffer<u32> stop_flag(1, dev_sched);
        stop_flag.fill(0);

        /*
            // To test to further optimize we can do something like:
            // (i tried and only got +30% which good but less than i expected)
            // A.K.A as something to do in another PR or never or I spam Claude on it

            auto range = sham::make_ndrange(group_size, (cnt + 3) / 4);

            <...>

            // fetch the u8 4 by 4, complete with 0x01 (s) if idx + 4 > cnt
            auto fetch_4i8 = [ptr = buf, cnt](u32 idx) -> u32 {
                if (idx + 4 <= cnt)
                    return *reinterpret_cast<const u32 *>(ptr + idx);

                u32 v = 0;
                u32 i = 0;
#pragma unroll
                for (; idx + i < cnt; ++i)
                    v |= u32(ptr[idx + i]) << (i * 8);
                if (i < 4)
                    v |= u32(0x01) << (i * 8);
                return v;
            };

            u32 gid = item.get_global_linear_id();

            // if there are
            bool local = (gid < cnt) ? (fetch_4i8(gid * 4) == 0x01010101) : true;
        */

        // TODO: switch to the check version when available
        auto range = sham::make_ndrange(group_size, cnt);

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf},
            sham::MultiRef{stop_flag},
            u32{1}, // TODO that when we have the new variant without it
            [=](u32, const T *buf, u32 *stop) {
                return [=](sycl::handler &cgh) {
                    cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                        auto grp = item.get_group();
                        u32 lid  = item.get_local_linear_id();

                        // Only the group leader reads the stop flag from device memory,
                        // then broadcast that single value to the rest of the group instead
                        // of every work-item issuing its own global memory load.
                        u32 stop_val = sycl::group_broadcast(grp, (lid == 0) ? *stop : u32{0}, 0);

                        // early exit the whole group if the flag is set
                        if (stop_val) {
                            return;
                        }

                        u32 gid = item.get_global_linear_id();

                        bool local = (gid < cnt) ? (buf[gid] != 0) : true;

                        // reduce in lid==0 the sum of local
                        bool result = sycl::all_of_group(grp, local);

                        if (lid == 0) {
                            // if there is a false we set the stop flag
                            if (!result && !(*stop)) {
                                sycl::atomic_ref<
                                    u32,
                                    sycl::memory_order_relaxed,
                                    sycl::memory_scope_device,
                                    sycl::access::address_space::global_space>
                                    atom(*stop);
                                atom |= 1_u32;
                            }
                        }
                    });
                };
            });

        return stop_flag.get_val_at_idx(0) == 0;
    }

} // namespace

namespace shamalgs::primitives::impl {

    /// Check all elements on host after copying the buffer back
    struct Host {
        static constexpr std::string_view variant_type_name = "host";
    };

    /// Check all elements via a sum reduction on device
    struct SumReduction {
        static constexpr std::string_view variant_type_name = "sum_reduction";
    };

    /// Check all elements via a sum reduction on device
    struct AtomicEarlyExit {
        static constexpr std::string_view variant_type_name = "atomic_early_exit";
        u32 group_size                                      = 256;

        /// Expose both group sizes worth benchmarking as separate default implementations,
        /// instead of only the default-constructed group_size = 256
        static std::vector<AtomicEarlyExit> variant_custom_defaults() {
            return {AtomicEarlyExit{64}, AtomicEarlyExit{256}};
        }
    };
} // namespace shamalgs::primitives::impl

template<>
struct shamalgs::ImplVariantParams<shamalgs::primitives::impl::AtomicEarlyExit> {
    static nlohmann::json to_json(const shamalgs::primitives::impl::AtomicEarlyExit &p) {
        return {{"group_size", p.group_size}};
    }
    static shamalgs::primitives::impl::AtomicEarlyExit from_json(const nlohmann::json &j) {
        shamalgs::primitives::impl::AtomicEarlyExit p{};
        if (j.contains("group_size")) {
            p.group_size = j.at("group_size").get<u32>();
        }
        return p;
    }
};

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        shamalgs::ImplVariantGlobal<Host, SumReduction, AtomicEarlyExit> is_all_true_impl;

        /// Get list of available is_all_true implementations, as config json strings
        std::vector<std::string> get_default_impl_list_is_all_true() {
            return is_all_true_impl.get_default_config_list();
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
                [&](impl::AtomicEarlyExit cfg) {
                    return is_all_true_early_group_exit(buf, cnt, cfg.group_size);
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
