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
 * @file flatten.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <type_traits>
#include <stdexcept>
#include <utility>

namespace shamalgs::primitives {

    /**
     * @brief Flatten a buffer of vector type into a buffer of scalar type.
     *
     * The buffer is flattened in row-major order, i.e. the first element of the first vector is
     * stored first, then the second element of the first vector, and so on.
     *
     * @param buffer The buffer to flatten.
     *
     * @return The flattened buffer.
     */
    template<class Tvec, sham::USMKindTarget target>
    inline sham::DeviceBuffer<typename shambase::VectorProperties<Tvec>::component_type, target>
    flatten_buffer(sham::DeviceBuffer<Tvec, target> &buffer) {

        using Tscal = typename shambase::VectorProperties<Tvec>::component_type;
        auto &sched = buffer.get_dev_scheduler_ptr();

        if constexpr (target == sham::USMKindTarget::device) {

            if constexpr (std::is_same_v<Tvec, Tscal>) {
                return buffer.copy();
            } else if constexpr (std::is_same_v<Tvec, sycl::vec<Tscal, 2>>) {

                sham::DeviceBuffer<Tscal, target> ret(buffer.get_size() * 2, sched);

                sham::EventList depends_list;
                const Tvec *ptr_src = buffer.get_read_access(depends_list);
                Tscal *ptr_dest     = ret.get_write_access(depends_list);

                sycl::event e = buffer.get_dev_scheduler().get_queue().submit(
                    depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(buffer.get_size(), [=](sycl::id<1> gid) {
                            Tvec tmp              = ptr_src[gid];
                            ptr_dest[gid * 2 + 0] = tmp[0];
                            ptr_dest[gid * 2 + 1] = tmp[1];
                        });
                    });

                ret.complete_event_state(e);
                buffer.complete_event_state(e);

                return ret;

            } else if constexpr (std::is_same_v<Tvec, sycl::vec<Tscal, 3>>) {

                sham::DeviceBuffer<Tscal, target> ret(buffer.get_size() * 3, sched);

                sham::EventList depends_list;
                const Tvec *ptr_src = buffer.get_read_access(depends_list);
                Tscal *ptr_dest     = ret.get_write_access(depends_list);

                sycl::event e = buffer.get_dev_scheduler().get_queue().submit(
                    depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(buffer.get_size(), [=](sycl::id<1> gid) {
                            Tvec tmp              = ptr_src[gid];
                            ptr_dest[gid * 3 + 0] = tmp[0];
                            ptr_dest[gid * 3 + 1] = tmp[1];
                            ptr_dest[gid * 3 + 2] = tmp[2];
                        });
                    });

                ret.complete_event_state(e);
                buffer.complete_event_state(e);
                return ret;

            } else {
                shambase::throw_unimplemented();
            }

        } else {
            shambase::throw_unimplemented();
        }
    }

    /**
     * @brief Unflatten a buffer that contains a flattened vector.
     *
     * @param buffer The buffer to unflatten
     *
     * @return A new buffer of type Tvec with the same size as the original buffer
     * divided by the number of components in the vector.
     *
     * @throws std::invalid_argument if the buffer has a size that is not a multiple
     * of the number of components in the vector.
     *
     * @throws std::runtime_error if the buffer is not a device buffer.
     */
    template<class Tvec, sham::USMKindTarget target>
    inline sham::DeviceBuffer<Tvec, target> unflatten_buffer(
        sham::DeviceBuffer<typename shambase::VectorProperties<Tvec>::component_type, target>
            &buffer) {

        using Tscal = typename shambase::VectorProperties<Tvec>::component_type;
        auto &sched = buffer.get_dev_scheduler_ptr();

        if constexpr (target == sham::USMKindTarget::device) {

            if constexpr (std::is_same_v<Tscal, Tvec>) {
                return buffer.copy();
            } else if constexpr (std::is_same_v<Tvec, sycl::vec<Tscal, 2>>) {

                if (buffer.get_size() % 2 != 0) {
                    shambase::throw_with_loc<std::invalid_argument>(
                        "The buffer must have an even number of elements");
                }

                sham::DeviceBuffer<Tvec, target> ret(buffer.get_size() / 2, sched);

                sham::EventList depends_list;
                const Tscal *ptr_src = buffer.get_read_access(depends_list);
                Tvec *ptr_dest       = ret.get_write_access(depends_list);

                sycl::event e = buffer.get_dev_scheduler().get_queue().submit(
                    depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(buffer.get_size() / 2, [=](sycl::id<1> gid) {
                            ptr_dest[gid] = Tvec{ptr_src[gid * 2 + 0], ptr_src[gid * 2 + 1]};
                        });
                    });

                ret.complete_event_state(e);
                buffer.complete_event_state(e);

                return ret;

            } else if constexpr (std::is_same_v<Tvec, sycl::vec<Tscal, 3>>) {

                if (buffer.get_size() % 3 != 0) {
                    shambase::throw_with_loc<std::invalid_argument>(
                        "The buffer must have a multiple of 3 elements");
                }

                sham::DeviceBuffer<Tvec, target> ret(buffer.get_size() / 3, sched);

                sham::EventList depends_list;
                const Tscal *ptr_src = buffer.get_read_access(depends_list);
                Tvec *ptr_dest       = ret.get_write_access(depends_list);

                sycl::event e = buffer.get_dev_scheduler().get_queue().submit(
                    depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(buffer.get_size() / 3, [=](sycl::id<1> gid) {
                            ptr_dest[gid] = Tvec{
                                ptr_src[gid * 3 + 0], ptr_src[gid * 3 + 1], ptr_src[gid * 3 + 2]};
                        });
                    });

                ret.complete_event_state(e);
                buffer.complete_event_state(e);

                return ret;
            } else {
                shambase::throw_unimplemented();
            }

        } else {
            shambase::throw_unimplemented();
        }
    }

} // namespace shamalgs
