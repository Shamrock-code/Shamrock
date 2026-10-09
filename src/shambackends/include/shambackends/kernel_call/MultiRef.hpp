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
 * @file MultiRef.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call/buffer_access_utils.hpp"

namespace sham {
    /**
     * @brief A class that references multiple buffers or similar objects.
     *
     * This class serves as a means to pass multiple buffers or objects with similar accessor
     * patterns to a kernel. It provides methods to obtain read and write access to these
     * entities and to complete their event state.
     *
     * A version of this class is also available for optional references to the buffers or similar
     * objects, @see MultiRefOpt.
     */
    template<class... Targ>
    struct MultiRef {
        /// A tuple of references to the buffers.
        using storage_t = std::tuple<Targ &...>;

        /// A tuple of references to the buffers.
        storage_t storage;

        /// Constructor
        MultiRef(Targ &...arg) : storage(arg...) {}

        /// Get a tuple of pointers to the data of the buffers, for reading. Register also the
        /// depedancies in depends_list.
        auto get_read_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::get_read_access(__a, depends_list)...);
                },
                storage);
        }

        /// Get a tuple of pointers to the data of the buffers, for writing. Register also the
        /// depedancies in depends_list.
        auto get_write_access(sham::EventList &depends_list) {
            __shamrock_stack_entry();
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::get_write_access(__a, depends_list)...);
                },
                storage);
        }

        /// Complete the event state of the buffers.
        /// @param e The SYCL event to register in the buffers.
        void complete_event_state(sycl::event e) {
            __shamrock_stack_entry();
            std::apply(
                [&](auto &...__in) {
                    ((details::complete_event_state(__in, e)), ...);
                },
                storage);
        }
    };
} // namespace sham
