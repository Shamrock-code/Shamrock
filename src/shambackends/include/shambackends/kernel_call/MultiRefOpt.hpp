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
 * @file MultiRefOpt.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call/buffer_access_utils.hpp"

namespace sham {

    /**
     * @brief Converts a reference to a given object into an optional reference wrapper.
     * @tparam T Type of the object to reference.
     * @param t Reference to the object.
     * @return An std::optional containing a std::reference_wrapper of the object.
     */
    template<class T>
    shambase::opt_ref<T> to_opt_ref(T &t) {
        return t;
    }

    /**
     * @brief Returns an empty optional containing a reference to a sham::DeviceBuffer<T>.
     * @details This function is useful when you want to pass an optional reference to a kernel
     * argument but you don't know if the argument is going to be used or not.
     * @return An empty std::optional containing a std::reference_wrapper of a
     * sham::DeviceBuffer<T>.
     */
    template<class T>
    auto empty_buf_ref() {
        return shambase::opt_ref<sham::DeviceBuffer<T>>{};
    }

    /**
     * @brief A variant of MultiRef for optional buffers.
     *
     * This class is equivalent to MultiRef but it allows optional buffers. Only DeviceBuffer are
     * supported as optional buffers.
     *
     * @see MultiRef
     */
    template<class... Targ>
    struct MultiRefOpt {
        /// A tuple of optional references to the buffers.
        using storage_t = std::tuple<shambase::opt_ref<Targ>...>;

        /// The tuple of optional references to the buffers.
        storage_t storage;

        /// Constructor from a tuple of optional references to the buffers.
        MultiRefOpt(shambase::opt_ref<Targ>... arg) : storage(arg...) {}

        /**
         * @brief Get a tuple of pointers to the data of the buffers, for reading.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the read
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_read_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::read_access_optional(__a, depends_list)...);
                },
                storage);
        }
        /**
         * @brief Get a tuple of pointers to the data of the buffers, for writing.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the write
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_write_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::write_access_optional(__a, depends_list)...);
                },
                storage);
        }

        /**
         * @brief Complete the event state of the buffers.
         * @details This function completes the event state of all the buffers in the
         * MultiRefOpt by registering the event `e` in all the buffers.
         *
         * @param e The SYCL event to register in the buffers.
         */
        void complete_event_state(sycl::event e) {
            __shamrock_stack_entry();
            std::apply(
                [&](auto &...__in) {
                    ((details::complete_state_optional(e, __in)), ...);
                },
                storage);
        }
    };

    namespace details {
        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper {
            /// The mapped type.
            using type = T;
        };

        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper<shambase::opt_ref<T>> {
            /// The mapped type.
            using type = T;
        };
    } // namespace details

    /// deduction guide to allow the MutliRefOpt to be build without the use of sham::to_opt_ref
    template<class... Targ>
    MultiRefOpt(Targ... arg) -> MultiRefOpt<typename details::mapper<Targ>::type...>;
} // namespace sham
