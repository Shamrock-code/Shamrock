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
#include "shamalgs/primitives/flatten.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/wrapper.hpp"
#include <type_traits>
#include <stdexcept>
#include <utility>

namespace shamalgs::collective {

    template<class T>
    inline T allreduce_one(T a, MPI_Op op, MPI_Comm comm) {
        T ret;
        shamcomm::mpi::Allreduce(&a, &ret, 1, get_mpi_type<T>(), op, comm);
        return ret;
    }

    template<class T, int n>
    inline sycl::vec<T, n> allreduce_one(sycl::vec<T, n> a, MPI_Op op, MPI_Comm comm) {
        sycl::vec<T, n> ret;
        if constexpr (n == 2) {
            shamcomm::mpi::Allreduce(&a.x(), &ret.x(), 1, get_mpi_type<T>(), op, comm);
            shamcomm::mpi::Allreduce(&a.y(), &ret.y(), 1, get_mpi_type<T>(), op, comm);
        } else if constexpr (n == 3) {
            shamcomm::mpi::Allreduce(&a.x(), &ret.x(), 1, get_mpi_type<T>(), op, comm);
            shamcomm::mpi::Allreduce(&a.y(), &ret.y(), 1, get_mpi_type<T>(), op, comm);
            shamcomm::mpi::Allreduce(&a.z(), &ret.z(), 1, get_mpi_type<T>(), op, comm);
        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>("unimplemented");
        }
        return ret;
    }

    template<class T>
    inline T allreduce_sum(T a) {
        return allreduce_one(a, MPI_SUM, MPI_COMM_WORLD);
    }

    template<class T>
    inline T allreduce_min(T a) {
        return allreduce_one(a, MPI_MIN, MPI_COMM_WORLD);
    }

    template<class T>
    inline T allreduce_max(T a) {
        return allreduce_one(a, MPI_MAX, MPI_COMM_WORLD);
    }

    template<class T>
    inline std::pair<T, T> allreduce_bounds(std::pair<T, T> bounds) {
        return {allreduce_min(bounds.first), allreduce_max(bounds.second)};
    }

    template<class T, sham::USMKindTarget target>
    inline void reduce_buffer_in_place_sum(sham::DeviceBuffer<T, target> &field, MPI_Comm comm) {

        if constexpr (shambase::VectorProperties<T>::dimension > 1) {
            auto flat = shamalgs::primitives::flatten_buffer(field);
            reduce_buffer_in_place_sum(flat, comm);
            field = shamalgs::primitives::unflatten_buffer<T, target>(flat);
        } else {

            if (field.get_size() > size_t(i32_max)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "MPI message are limited to i32_max in size");
            }

            if constexpr (target == sham::device) {

                if (field.get_dev_scheduler().use_direct_comm()) {
                    sham::EventList depends_list;
                    T *ptr = field.get_write_access(depends_list);

                    depends_list.wait_and_throw();

                    shamcomm::mpi::Allreduce(
                        MPI_IN_PLACE, ptr, field.get_size(), get_mpi_type<T>(), MPI_SUM, comm);

                    field.complete_event_state(sycl::event{});
                } else {
                    sham::DeviceBuffer<T, sham::host> field_host
                        = field.template copy_to<sham::host>();
                    reduce_buffer_in_place_sum(field_host, comm);
                    field.copy_from(field_host);
                }

            } else if (target == sham::host) {

                sham::EventList depends_list;
                T *ptr = field.get_write_access(depends_list);

                depends_list.wait_and_throw();

                shamcomm::mpi::Allreduce(
                    MPI_IN_PLACE, ptr, field.get_size(), get_mpi_type<T>(), MPI_SUM, comm);

                field.complete_event_state(sycl::event{});
            } else {
                shambase::throw_unimplemented();
            }
        }
    }

} // namespace shamalgs::collective
