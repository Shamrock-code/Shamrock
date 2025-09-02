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
 * @file wrapper.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shamcomm/mpi.hpp"
#include <string>

namespace shamcomm::mpi {

    /// Register a timer value
    void register_time(std::string timername, f64 time);

    /// get a timer value
    f64 get_timer(std::string timername);

    /// MPI wrapper for MPI_Allreduce
    void Allreduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm);

    /// MPI wrapper for MPI_Allgather
    void Allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

    /// MPI wrapper for MPI_Allgatherv
    void Allgatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        MPI_Comm comm);

    /// MPI wrapper for MPI_Isend
    void Isend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request);

    /// MPI wrapper for MPI_Irecv
    void Irecv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Request *request);

    /// MPI wrapper for MPI_Exscan
    void Exscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm);

    /// MPI wrapper for MPI_Wait
    void Wait(MPI_Request *request, MPI_Status *status);

    /// MPI wrapper for MPI_Waitall
    void Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

    /// MPI wrapper for MPI_Barrier
    void Barrier(MPI_Comm comm);

    /// MPI wrapper for MPI_Probe
    void Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);

    /// MPI wrapper for MPI_Recv
    void Recv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Status *status);

    /// MPI wrapper for MPI_Get_count
    void Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);

    /// MPI wrapper for MPI_Send
    void Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

    /// MPI wrapper for MPI_File_set_view
    void File_set_view(
        MPI_File fh,
        MPI_Offset disp,
        MPI_Datatype etype,
        MPI_Datatype filetype,
        const char *datarep,
        MPI_Info info);

    /// MPI wrapper for MPI_Type_size
    void Type_size(MPI_Datatype type, int *size);

    /// MPI wrapper for MPI_File_write_all
    void File_write_all(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    /// MPI wrapper for MPI_File_write
    void File_write(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    /// MPI wrapper for MPI_File_read
    void File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    /// MPI wrapper for MPI_File_write_at
    void File_write_at(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status);

    /// MPI wrapper for MPI_File_read_at
    void File_read_at(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status);

    /// MPI wrapper for MPI_File_close
    void File_close(MPI_File *fh);

    /// MPI wrapper for MPI_File_open
    void File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

    /// MPI wrapper for MPI_Test
    void Test(MPI_Request *request, int *flag, MPI_Status *status);

    /// MPI wrapper for MPI_Gather
    void Gather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm);

    /// MPI wrapper for MPI_Gatherv
    void Gatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm);

} // namespace shamcomm::mpi
