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
 * @file buffer_access_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/optional.hpp"
#include "shambackends/DeviceBuffer.hpp"

namespace sham::details {

    /**
     * @brief Get a pointer to the data of an optional device buffer, for reading.
     * @details If the optional is empty, a null pointer is returned. Otherwise, the read
     * access of the buffer is requested and the depends_list is updated accordingly.
     *
     * @param buffer An optional holding a reference to the device buffer.
     * @param depends_list The list of events to wait for.
     * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
     */
    template<class T>
    const T *read_access_optional(
        shambase::opt_ref<sham::DeviceBuffer<T>> buffer, sham::EventList &depends_list) {
        if (!buffer.has_value()) {
            return nullptr;
        } else {
            return buffer.value().get().get_read_access(depends_list);
        }
    }

    /**
     * @brief Get a pointer to the data of an optional device buffer, for writing.
     * @details If the optional is empty, a null pointer is returned. Otherwise, the write
     * access of the buffer is requested and the depends_list is updated accordingly.
     *
     * @param buffer An optional holding a reference to the device buffer.
     * @param depends_list The list of events to wait for.
     * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
     */
    template<class T>
    T *write_access_optional(
        shambase::opt_ref<sham::DeviceBuffer<T>> buffer, sham::EventList &depends_list) {
        if (!buffer.has_value()) {
            return nullptr;
        } else {
            return buffer.value().get().get_write_access(depends_list);
        }
    }

    /**
     * @brief Complete the event state of an optional device buffer.
     * @details If the optional is empty, nothing is done. Otherwise, the event state of the
     * buffer is completed with the given event.
     */
    template<class T>
    void complete_state_optional(sycl::event e, shambase::opt_ref<T> buffer) {
        if (buffer.has_value()) {
            buffer.value().get().complete_event_state(e);
        }
    }

    template<class Obj>
    inline auto get_read_access(Obj &o, sham::EventList &depends_list) {
        return o.get_read_access(depends_list);
    }

    template<class Obj>
    inline auto get_write_access(Obj &o, sham::EventList &depends_list) {
        return o.get_write_access(depends_list);
    }
    template<class Obj>
    inline auto complete_event_state(Obj &o, sycl::event e) {
        return o.complete_event_state(e);
    }

    template<class Obj>
    inline auto get_read_access(std::reference_wrapper<Obj> &o, sham::EventList &depends_list) {
        return o.get().get_read_access(depends_list);
    }

    template<class Obj>
    inline auto get_write_access(std::reference_wrapper<Obj> &o, sham::EventList &depends_list) {
        return o.get().get_write_access(depends_list);
    }
    template<class Obj>
    inline auto complete_event_state(std::reference_wrapper<Obj> &o, sycl::event e) {
        return o.get().complete_event_state(e);
    }

} // namespace sham::details
