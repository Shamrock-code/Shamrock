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

} // namespace shamalgs::collective
