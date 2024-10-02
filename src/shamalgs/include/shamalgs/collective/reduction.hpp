// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file reduction.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/flatten.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include <type_traits>
#include <stdexcept>
#include <utility>

namespace shamalgs::collective {

    template<class T>
    inline T allreduce_one(T a, MPI_Op op, MPI_Comm comm) {
        T ret;
        MPICHECK(MPI_Allreduce(&a, &ret, 1, get_mpi_type<T>(), op, comm));
        return ret;
    }

    template<class T, int n>
    inline sycl::vec<T, n> allreduce_one(sycl::vec<T, n> a, MPI_Op op, MPI_Comm comm) {
        sycl::vec<T, n> ret;
        if constexpr (n == 2) {
            MPICHECK(MPI_Allreduce(&a.x(), &ret.x(), 1, get_mpi_type<T>(), op, comm));
            MPICHECK(MPI_Allreduce(&a.y(), &ret.y(), 1, get_mpi_type<T>(), op, comm));
        } else if constexpr (n == 3) {
            MPICHECK(MPI_Allreduce(&a.x(), &ret.x(), 1, get_mpi_type<T>(), op, comm));
            MPICHECK(MPI_Allreduce(&a.y(), &ret.y(), 1, get_mpi_type<T>(), op, comm));
            MPICHECK(MPI_Allreduce(&a.z(), &ret.z(), 1, get_mpi_type<T>(), op, comm));
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
            auto flat = shamalgs::flatten_buffer(field);
            reduce_buffer_in_place_sum(flat, comm);
            field = shamalgs::unflatten_buffer<T, target>(flat);
        } else {

            if (field.get_size() > size_t(i32_max)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "MPI message are limited to i32_max in size");
            }

            if constexpr (target == sham::device) {

                if (field.get_dev_scheduler().use_direct_comm()) {
                    std::vector<sycl::event> depends_list;
                    T *ptr = field.get_write_access(depends_list);

                    for (auto &e : depends_list) {
                        e.wait_and_throw();
                    }

                    MPICHECK(MPI_Allreduce(
                        MPI_IN_PLACE, ptr, field.get_size(), get_mpi_type<T>(), MPI_SUM, comm));

                    field.complete_event_state({});
                } else {
                    sham::DeviceBuffer<T, sham::host> field_host
                        = field.template copy_to<sham::host>();
                    reduce_buffer_in_place_sum(field_host, comm);
                    field.copy_from(field_host);
                }

            } else if (target == sham::host) {

                std::vector<sycl::event> depends_list;
                T *ptr = field.get_write_access(depends_list);

                for (auto &e : depends_list) {
                    e.wait_and_throw();
                }

                MPICHECK(MPI_Allreduce(
                    MPI_IN_PLACE, ptr, field.get_size(), get_mpi_type<T>(), MPI_SUM, comm));

                field.complete_event_state({});
            } else {
                shambase::throw_unimplemented();
            }
        }
    }

} // namespace shamalgs::collective
