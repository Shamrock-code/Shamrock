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
 * @file BufferMirror.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"

namespace sham {

    /**
     * @brief A class template for creating a mirrored buffer
     *
     * This class provides a mechanism to create a mirrored buffer of a specific
     * DeviceBuffer that can be accessed like a normal pointer.
     *
     * Exemple of usage :
     * \code{.cpp}
     * {
     *     auto acc = f.get_buf().template mirror_to<sham::host>();
     *     auto acc_xyz = xyz.get_buf().template mirror_to<sham::host>();
     *
     *     for (u32 i = 0; i < f.size(); i++) {
     *         Tvec position = acc_xyz[i];
     *         acc[i] =function(position);
     *     }
     *
     * }
     * \endcode
     *
     * @tparam T The type of elements in the buffer
     * @tparam target The USM kind target for the mirror buffer
     * @tparam orgin_target The original USM kind target of the mirrored buffer
     */
    template<class T, USMKindTarget target, USMKindTarget orgin_target>
    class BufferMirror {

        DeviceBuffer<T, orgin_target> &mirrored_buffer; ///< Reference to the original buffer

        DeviceBuffer<T, target> mirror; ///< The mirrored buffer
        T *ptr_mirror;                  ///< Pointer to the mirrored data

        public:
        /**
         * @brief Constructs a BufferMirror
         *
         * Initializes the mirrored buffer of the original data and retain a reference to the
         * original buffer
         *
         * @param mirrored_buffer The original buffer to be mirrored
         */
        BufferMirror(DeviceBuffer<T, orgin_target> &mirrored_buffer)
            : mirrored_buffer(mirrored_buffer), mirror(mirrored_buffer.template copy_to<target>()) {
            sham::EventList depends_list;
            ptr_mirror = mirror.get_write_access(depends_list);
            depends_list.wait();
            mirror.complete_event_state(sycl::event{});
        }

        BufferMirror(const BufferMirror &)            = delete; //< Please don't
        BufferMirror(BufferMirror &&)                 = delete; //< Please don't
        BufferMirror &operator=(const BufferMirror &) = delete; //< Please don't
        BufferMirror &operator=(BufferMirror &&)      = delete; //< Please don't

        /**
         * @brief Provides access to the mirrored data
         *
         * @return Pointer to the mirrored data
         */
        T *data() const { return ptr_mirror; }

        /**
         * @brief Returns the size of the mirrored buffer
         *
         * @return Size of the mirrored buffer
         */
        u32 size() const { return mirrored_buffer.size(); }

        /**
         * @brief Accesses the mirrored data at a given index
         *
         * @param i Index of the data to access
         * @return Reference to the data at the given index
         */
        T &operator[](u32 i) const { return ptr_mirror[i]; }

        /**
         * @brief Destructor
         *
         * Copies the data from the mirror back to the original buffer.
         * We must wait for the copy to complete before the mirror is destroyed.
         */
        ~BufferMirror() {
            mirrored_buffer.copy_from(mirror);
            mirrored_buffer.synchronize();
        }
    };

} // namespace sham
