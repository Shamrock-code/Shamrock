// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file wrapper.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/profiling/profiling.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <unordered_map>
#include <array>

namespace {

    std::unordered_map<std::string, f64> mpi_timers;

} // namespace

namespace shamcomm::mpi {
    void register_time(std::string timername, f64 time) {
        mpi_timers[timername] += time;
        mpi_timers["total"] += time;

        if (shambase::profiling::is_profiling_enabled()) {
            auto wtime = shambase::details::get_wtime();
            shambase::profiling::register_counter_val(timername, wtime, mpi_timers[timername]);
            shambase::profiling::register_counter_val("total MPi time", wtime, mpi_timers["total"]);
        }
    }

    f64 get_timer(std::string timername) { return mpi_timers[timername]; }

} // namespace shamcomm::mpi

namespace {

    template<class Func>
    inline void wrap_profiling(std::string timername, Func &&f) {
        f64 tstart;
        tstart = shambase::details::get_wtime();
        f();
        shamcomm::mpi::register_time(timername, shambase::details::get_wtime() - tstart);
    }

} // namespace

namespace shamcomm::mpi {

    void check_tag_value(i32 tag) {
        if (tag > mpi_max_tag_value()) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "mpi_max_tag_value ({}) exceeded with tag {}", mpi_max_tag_value(), tag));
        }
    }

    void Isend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        StackEntry stack_loc{};

        check_tag_value(tag);

        wrap_profiling("MPI_Isend", [&]() {
            MPICHECK(MPI_Isend(buf, count, datatype, dest, tag, comm, request));
        });
    }

    void Irecv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        StackEntry stack_loc{};

        check_tag_value(tag);

        wrap_profiling("MPI_Irecv", [&]() {
            MPICHECK(MPI_Irecv(buf, count, datatype, source, tag, comm, request));
        });
    }

    void Allreduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        StackEntry stack_loc{};

        wrap_profiling("MPI_Allreduce", [&]() {
            MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
        });
    }

    void Allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        StackEntry stack_loc{};

        wrap_profiling("MPI_Allgather", [&]() {
            MPICHECK(
                MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
        });
    }

    void Allgatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        StackEntry stack_loc{};

        wrap_profiling("MPI_Allgatherv", [&]() {
            MPICHECK(MPI_Allgatherv(
                sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm));
        });
    }

    void Exscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        StackEntry stack_loc{};

        wrap_profiling("MPI_Exscan", [&]() {
            MPICHECK(MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm));
        });
    }

    void Wait(MPI_Request *request, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Wait", [&]() {
            MPICHECK(MPI_Wait(request, status));
        });
    }

    void Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Waitall", [&]() {
            MPICHECK(MPI_Waitall(count, array_of_requests, array_of_statuses));
        });
    }

    void Barrier(MPI_Comm comm) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Barrier", [&]() {
            MPICHECK(MPI_Barrier(comm));
        });
    }

    void Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Probe", [&]() {
            MPICHECK(MPI_Probe(source, tag, comm, status));
        });
    }

    void Recv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Recv", [&]() {
            MPICHECK(MPI_Recv(buf, count, datatype, source, tag, comm, status));
        });
    }

    void Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Get_count", [&]() {
            MPICHECK(MPI_Get_count(status, datatype, count));
        });
    }

    void Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Send", [&]() {
            MPICHECK(MPI_Send(buf, count, datatype, dest, tag, comm));
        });
    }

    void File_set_view(
        MPI_File fh,
        MPI_Offset disp,
        MPI_Datatype etype,
        MPI_Datatype filetype,
        const char *datarep,
        MPI_Info info) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_set_view", [&]() {
            MPICHECK(MPI_File_set_view(fh, disp, etype, filetype, datarep, info));
        });
    }

    void Type_size(MPI_Datatype type, int *size) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Type_size", [&]() {
            MPICHECK(MPI_Type_size(type, size));
        });
    }

    void File_write_all(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_write_all", [&]() {
            MPICHECK(MPI_File_write_all(fh, buf, count, datatype, status));
        });
    }

    void
    File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_write", [&]() {
            MPICHECK(MPI_File_write(fh, buf, count, datatype, status));
        });
    }

    void File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_read", [&]() {
            MPICHECK(MPI_File_read(fh, buf, count, datatype, status));
        });
    }

    void File_write_at(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_write_at", [&]() {
            MPICHECK(MPI_File_write_at(fh, offset, buf, count, datatype, status));
        });
    }

    void File_read_at(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_read_at", [&]() {
            MPICHECK(MPI_File_read_at(fh, offset, buf, count, datatype, status));
        });
    }

    void File_close(MPI_File *fh) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_close", [&]() {
            MPICHECK(MPI_File_close(fh));
        });
    }

    void File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_File_open", [&]() {
            MPICHECK(MPI_File_open(comm, filename, amode, info, fh));
        });
    }

    void Test(MPI_Request *request, int *flag, MPI_Status *status) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Test", [&]() {
            MPICHECK(MPI_Test(request, flag, status));
        });
    }

    void Gather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Gather", [&]() {
            MPICHECK(
                MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm));
        });
    }

    void Gatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm) {
        StackEntry stack_loc{};
        wrap_profiling("MPI_Gatherv", [&]() {
            MPICHECK(MPI_Gatherv(
                sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm));
        });
    }

} // namespace shamcomm::mpi
