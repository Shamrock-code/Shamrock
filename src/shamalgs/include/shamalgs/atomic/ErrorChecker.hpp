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
 * @file ErrorChecker.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::atomic {

    /**
     * @brief A utility class to check for errors on device, using a single uint
     *        to store all the error flags.
     *
     * @details
     * This class provides a way to check for errors on device,
     * using a single uint to store all the error flags.
     * The error flags are packed in the uint and can be checked
     * using the `is_flag_on` function.
     *
     * @code {.cpp}
     * auto sched = shamsys::instance::get_compute_scheduler_ptr();
     *
     * enum ErrorCodes : u32 {
     *     Flag1 = 1 << 0,
     *     Flag2 = 1 << 1,
     *     Flag3 = 1 << 2,
     * };
     *
     * shamalgs::atomic::ErrorCheckerFlags error_util(sched);
     *
     * sham::kernel_call(
     *     sched->get_queue(),
     *     sham::MultiRef{},
     *     sham::MultiRef{error_util},
     *     100,
     *     [](u32 i, auto error_util) {
     *         if (i == 2) {
     *             error_util.set_flag_on(Flag1);
     *         }
     *         if (i == 23 || i == 101) {
     *             error_util.set_flag_on(Flag2);
     *         }
     *     });
     *
     * u32 precondition_error = error_util.get_output();
     *
     * bool Flag_1_on = shambase::is_flag_on<Flag1>(precondition_error);
     * bool Flag_2_on = shambase::is_flag_on<Flag2>(precondition_error);
     * bool Flag_3_on = shambase::is_flag_on<Flag3>(precondition_error);
     * @endcode
     */
    struct ErrorCheckerFlags {

        /// The buffer used to store the error flag
        sham::DeviceBuffer<u32> buf_err;

        /// Constructor
        ErrorCheckerFlags(sham::DeviceScheduler_ptr sched) : buf_err(1, sched) { buf_err.fill(0); }

        /// A struct to access the pointer associated to the buffer
        struct accessed {
            u32 *ptr; ///< The pointer to the buffer

            /// Set a flag on the error flags buffer atomically.
            void set_flag_on(u32 flag_val) const {
                sycl::atomic_ref<
                    u32,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    atom(*ptr);
                atom |= flag_val;
            }
        };

        /// Get a write access to the buffer
        accessed get_write_access(sham::EventList &depends_list) {
            return accessed{buf_err.get_write_access(depends_list)};
        }

        /// Complete the event state
        void complete_event_state(sycl::event e) { buf_err.complete_event_state(e); }

        /// Get the resulting error flag
        u32 get_output() { return buf_err.copy_to_stdvec().at(0); }
    };

    /**
     * @brief This class is used to check for errors in kernels. It is composed of a buffer of u32
     * that is used to store the error counts.
     *
     * The class provides a method to count errors for multiple flags. The flags are set by doing a
     * fetch add operation on the corresponding element of the buffer. The element is chosen by the
     * argument of the `set_error` method. The value of the element is incremented by 1 each time
     * the `set_error` method is called.
     *
     *
     * @code {.cpp}
     * auto sched = shamsys::instance::get_compute_scheduler_ptr();
     *
     * shamalgs::atomic::ErrorCheckCounter error_util(sched, 3);
     *
     * sham::kernel_call(
     *     sched->get_queue(),
     *     sham::MultiRef{},
     *     sham::MultiRef{error_util},
     *     100,
     *     [](u32 i, auto error_util) {
     *         if (i == 2) {
     *             error_util.set_error(0);
     *         }
     *         if (i == 23 || i == 101) {
     *             error_util.set_error(1);
     *         }
     *     });
     *
     * auto precondition_error = error_util.get_outputs();
     * @endcode
     *
     */
    struct ErrorCheckCounter {

        /// The buffer used to store the error counts
        sham::DeviceBuffer<u32> buf_err;

        /// Constructor
        ErrorCheckCounter(sham::DeviceScheduler_ptr sched, u32 error_counter)
            : buf_err(error_counter, sched) {
            buf_err.fill(0_u32);
        }

        /// A struct to access the pointer associated to the buffer
        struct accessed {
            u32 *ptr; ///< The pointer to the buffer

            /// Increments the error count associated to the given id in the buffer.
            void set_error(u32 id) const {
                sycl::atomic_ref<
                    u32,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    atom(ptr[id]);
                atom.fetch_add(1_u32);
            }
        };

        /// Get a write access to the buffer
        accessed get_write_access(sham::EventList &depends_list) {
            return accessed{buf_err.get_write_access(depends_list)};
        }

        /// Complete the event state
        void complete_event_state(sycl::event e) { buf_err.complete_event_state(e); }

        /// Get the resulting error counts
        std::vector<u32> get_outputs() { return buf_err.copy_to_stdvec(); }
    };

} // namespace shamalgs::atomic
