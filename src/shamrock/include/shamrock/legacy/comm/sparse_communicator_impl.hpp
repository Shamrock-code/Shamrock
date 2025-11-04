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
 * @file sparse_communicator_impl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/io/logs.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include <vector>

// TODO move to a contained module

namespace impl {
    template<class T>
    inline void vector_isend(
        std::vector<T> &p,
        std::vector<MPI_Request> &rq_lst,
        i32 rank_dest,
        i32 tag,
        MPI_Comm comm) {
        rq_lst.resize(rq_lst.size() + 1);
        shamcomm::mpi::Isend(
            p.data(),
            p.size(),
            get_mpi_type<T>(),
            rank_dest,
            tag,
            comm,
            &rq_lst[rq_lst.size() - 1]);
    }

    template<class T>
    inline void vector_irecv(
        std::vector<T> &pdat,
        std::vector<MPI_Request> &rq_lst,
        i32 rank_source,
        i32 tag,
        MPI_Comm comm) {
        MPI_Status st;
        i32 cnt;
        int i = shamcomm::mpi::Probe(rank_source, tag, comm, &st);
        shamcomm::mpi::Get_count(&st, get_mpi_type<T>(), &cnt);
        rq_lst.resize(rq_lst.size() + 1);
        pdat.resize(cnt);
        shamcomm::mpi::Irecv(
            pdat.data(),
            cnt,
            get_mpi_type<T>(),
            rank_source,
            tag,
            comm,
            &rq_lst[rq_lst.size() - 1]);
    }
} // namespace impl
