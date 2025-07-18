// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sycl_mpi_interop.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * @version 0.1
 * @date 2022-03-14
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "log.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"

#define XMAC_SYCLMPI_TYPE_ENABLED                                                                  \
    X(f32)                                                                                         \
    X(f32_2)                                                                                       \
    X(f32_3)                                                                                       \
    X(f32_4)                                                                                       \
    X(f32_8)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64)                                                                                         \
    X(f64_2)                                                                                       \
    X(f64_3)                                                                                       \
    X(f64_4)                                                                                       \
    X(f64_8)                                                                                       \
    X(f64_16)                                                                                      \
    X(u8)                                                                                          \
    X(u32)                                                                                         \
    X(u32_3)                                                                                       \
    X(u16_3)                                                                                       \
    X(u64)                                                                                         \
    X(u64_3)                                                                                       \
    X(i64_3)                                                                                       \
    X(i64)

namespace mpi_sycl_interop {

    enum comm_type { CopyToHost, DirectGPU };
    enum op_type { Send, Recv_Probe };

    extern comm_type current_mode;

    template<class T>
    struct BufferMpiRequest {

        static constexpr bool is_in_type_list =
#define X(args) std::is_same<T, args>::value ||
            XMAC_SYCLMPI_TYPE_ENABLED false
#undef X
            ;

        static_assert(
            is_in_type_list,
            "BufferMpiRequest must be one of those types : "

#define X(args) #args " "
            XMAC_SYCLMPI_TYPE_ENABLED
#undef X
        );

        MPI_Request mpi_rq;
        comm_type comm_mode;
        op_type comm_op;
        T *comm_ptr;
        u32 comm_sz;
        std::unique_ptr<sycl::buffer<T>> &sycl_buf;

        BufferMpiRequest<T>(
            std::unique_ptr<sycl::buffer<T>> &sycl_buf,
            comm_type comm_mode,
            op_type comm_op,
            u32 comm_sz);

        inline T *get_mpi_ptr() { return comm_ptr; }

        void finalize();
    };

    template<class T>
    inline u64 isend(
        std::unique_ptr<sycl::buffer<T>> &p,
        const u32 &size_comm,
        std::vector<BufferMpiRequest<T>> &rq_lst,
        i32 rank_dest,
        i32 tag,
        MPI_Comm comm) {

        rq_lst.push_back(BufferMpiRequest<T>(p, current_mode, Send, size_comm));

        u32 rq_index = rq_lst.size() - 1;

        auto &rq = rq_lst[rq_index];

        shamcomm::mpi::Isend(
            rq.get_mpi_ptr(),
            size_comm,
            get_mpi_type<T>(),
            rank_dest,
            tag,
            comm,
            &(rq_lst[rq_index].mpi_rq));

        return sizeof(T) * size_comm;
    }

    template<class T>
    inline u64 irecv(
        std::unique_ptr<sycl::buffer<T>> &p,
        const u32 &size_comm,
        std::vector<BufferMpiRequest<T>> &rq_lst,
        i32 rank_source,
        i32 tag,
        MPI_Comm comm) {

        rq_lst.push_back(BufferMpiRequest<T>(p, current_mode, Recv_Probe, size_comm));

        u32 rq_index = rq_lst.size() - 1;

        auto &rq = rq_lst[rq_index];

        shamcomm::mpi::Irecv(
            rq.get_mpi_ptr(),
            size_comm,
            get_mpi_type<T>(),
            rank_source,
            tag,
            comm,
            &(rq_lst[rq_index].mpi_rq));

        return sizeof(T) * size_comm;
    }

    template<class T>
    inline u64 irecv_probe(
        std::unique_ptr<sycl::buffer<T>> &p,
        std::vector<BufferMpiRequest<T>> &rq_lst,
        i32 rank_source,
        i32 tag,
        MPI_Comm comm) {
        MPI_Status st;
        i32 cnt;
        shamcomm::mpi::Probe(rank_source, tag, comm, &st);
        shamcomm::mpi::Get_count(&st, get_mpi_type<T>(), &cnt);

        u32 len = cnt;

        return irecv(p, len, rq_lst, rank_source, tag, comm);
    }

    template<class T>
    inline std::vector<MPI_Request> get_rqs(std::vector<BufferMpiRequest<T>> &rq_lst) {
        std::vector<MPI_Request> addrs;

        for (auto a : rq_lst) {
            addrs.push_back(a.mpi_rq);
        }

        return addrs;
    }

    template<class T>
    inline void waitall(std::vector<BufferMpiRequest<T>> &rq_lst) {
        std::vector<MPI_Request> addrs;

        for (auto a : rq_lst) {
            addrs.push_back(a.mpi_rq);
        }

        std::vector<MPI_Status> st_lst(addrs.size());
        shamcomm::mpi::Waitall(addrs.size(), addrs.data(), st_lst.data());

        for (auto a : rq_lst) {
            a.finalize();
        }
    }

    template<class T>
    inline void file_write(MPI_File fh, std::unique_ptr<sycl::buffer<T>> &p, const u32 &size_comm) {
        MPI_Status st;

        BufferMpiRequest<T> rq(p, current_mode, Send, size_comm);

        shamcomm::mpi::File_write(fh, rq.get_mpi_ptr(), size_comm, get_mpi_type<T>(), &st);

        rq.finalize();
    }

} // namespace mpi_sycl_interop

namespace impl::copy_to_host {

    namespace send {
        template<class T>
        T *init(const std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz);

        template<class T>
        void finalize(T *comm_ptr);
    } // namespace send

    namespace recv {
        template<class T>
        T *init(u32 comm_sz);

        template<class T>
        void finalize(const std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz);
    } // namespace recv

} // namespace impl::copy_to_host

namespace impl::directgpu {

    namespace send {
        template<class T>
        T *init(const std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz);

        template<class T>
        void finalize(T *comm_ptr);
    } // namespace send

    namespace recv {
        template<class T>
        T *init(u32 comm_sz);

        template<class T>
        void finalize(const std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz);
    } // namespace recv

} // namespace impl::directgpu
