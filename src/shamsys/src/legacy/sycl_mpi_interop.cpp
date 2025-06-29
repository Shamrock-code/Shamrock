// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sycl_mpi_interop.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"

namespace impl::copy_to_host {

    // using namespace mpi_sycl_interop;

    namespace send {
        template<class T>
        T *init(const std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz) {

            using namespace shamsys::instance;

            T *comm_ptr = sycl::malloc_host<T>(comm_sz, get_compute_queue());
            get_compute_queue().wait();
            shamlog_debug_sycl_ln(
                "PatchDataField MPI Comm",
                "sycl::malloc_host",
                comm_sz,
                "->",
                reinterpret_cast<void *>(comm_ptr));

            if (comm_sz > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                {
                    sycl::host_accessor acc{shambase::get_check_ref(buf), sycl::read_only};

                    const T *src = acc.get_pointer();
                    T *dest      = comm_ptr;

                    std::memcpy(dest, src, sizeof(T) * comm_sz);
                }

            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

#define X(_t) template _t *init<_t>(const std::unique_ptr<sycl::buffer<_t>> &buf, u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X

        template<class T>
        void finalize(T *comm_ptr) {

            using namespace shamsys::instance;

            shamlog_debug_sycl_ln(
                "PatchDataField MPI Comm", "sycl::free", reinterpret_cast<void *>(comm_ptr));

            sycl::free(comm_ptr, get_compute_queue());
        }

#define X(_t) template void finalize(_t *comm_ptr);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X
    } // namespace send

    namespace recv {
        template<class T>
        T *init(u32 comm_sz) {

            using namespace shamsys::instance;

            T *comm_ptr = sycl::malloc_host<T>(comm_sz, shamsys::instance::get_compute_queue());

            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host", comm_sz);

            return comm_ptr;
        };

#define X(_t) template _t *init(u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X
        template<class T>
        void finalize(const std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz) {

            if (comm_sz > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                {
                    sycl::host_accessor acc{
                        shambase::get_check_ref(buf), sycl::write_only, sycl::no_init};

                    const T *src = comm_ptr;
                    T *dest      = acc.get_pointer();

                    std::memcpy(dest, src, sizeof(T) * comm_sz);
                }

            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            shamlog_debug_sycl_ln(
                "PatchDataField MPI Comm", "sycl::free", reinterpret_cast<void *>(comm_ptr));

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }

#define X(_t)                                                                                      \
    template void finalize(const std::unique_ptr<sycl::buffer<_t>> &buf, _t *comm_ptr, u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X
    } // namespace recv

} // namespace impl::copy_to_host

namespace impl::directgpu {

    using namespace mpi_sycl_interop;

    namespace send {
        template<class T>
        T *init(const std::unique_ptr<sycl::buffer<T>> &buf, u32 comm_sz) {

            T *comm_ptr = sycl::malloc_device<T>(comm_sz, shamsys::instance::get_compute_queue());
            shamlog_debug_sycl_ln(
                "PatchDataField MPI Comm", "sycl::malloc_device", comm_sz, "->", comm_ptr);

            if (comm_sz > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                auto ker_copy
                    = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                          sycl::accessor acc{*buf, cgh, sycl::read_only};

                          T *ptr = comm_ptr;

                          cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) {
                              ptr[item.get_linear_id()] = acc[item];
                          });
                      });

                ker_copy.wait();
            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
            }

            return comm_ptr;
        }

#define X(_t) template _t *init<_t>(const std::unique_ptr<sycl::buffer<_t>> &buf, u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X

        template<class T>
        void finalize(T *comm_ptr) {
            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }

#define X(_t) template void finalize(_t *comm_ptr);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X
    } // namespace send

    namespace recv {
        template<class T>
        T *init(u32 comm_sz) {
            T *comm_ptr = sycl::malloc_device<T>(comm_sz, shamsys::instance::get_compute_queue());

            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_device", comm_sz);

            return comm_ptr;
        };

#define X(_t) template _t *init(u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X
        template<class T>
        void finalize(const std::unique_ptr<sycl::buffer<T>> &buf, T *comm_ptr, u32 comm_sz) {

            if (comm_sz > 0) {
                shamlog_debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                auto ker_copy
                    = shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                          sycl::accessor acc{*buf, cgh, sycl::write_only};

                          T *ptr = comm_ptr;

                          cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item) {
                              acc[item] = ptr[item.get_linear_id()];
                          });
                      });

                ker_copy.wait();
            } else {
                shamlog_debug_sycl_ln(
                    "PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
            }

            shamlog_debug_sycl_ln("PatchDataField MPI Comm", "sycl::free", comm_ptr);

            sycl::free(comm_ptr, shamsys::instance::get_compute_queue());
        }
#define X(_t)                                                                                      \
    template void finalize(const std::unique_ptr<sycl::buffer<_t>> &buf, _t *comm_ptr, u32 comm_sz);
        XMAC_SYCLMPI_TYPE_ENABLED
#undef X

    } // namespace recv

} // namespace impl::directgpu

namespace mpi_sycl_interop {

    comm_type current_mode = CopyToHost;

    template<class T>
    BufferMpiRequest<T>::BufferMpiRequest(
        std::unique_ptr<sycl::buffer<T>> &sycl_buf,
        comm_type comm_mode,
        op_type comm_op,
        u32 comm_sz)
        : comm_mode(comm_mode), comm_op(comm_op), comm_sz(comm_sz), sycl_buf(sycl_buf) {

        shamlog_debug_mpi_ln(
            "PatchDataField MPI Comm",
            "starting mpi sycl comm ",
            comm_sz,
            int(comm_op),
            int(comm_mode));

        if (comm_mode == CopyToHost && comm_op == Send) {

            comm_ptr = impl::copy_to_host::send::init<T>(sycl_buf, comm_sz);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            comm_ptr = impl::copy_to_host::recv::init<T>(comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            comm_ptr = impl::directgpu::send::init<T>(sycl_buf, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            comm_ptr = impl::directgpu::recv::init<T>(comm_sz);

        } else {
            logger::err_ln(
                "PatchDataField MPI Comm",
                "communication mode & op combination not implemented :",
                int(comm_mode),
                int(comm_op));
        }
    }

    template<class T>
    void BufferMpiRequest<T>::finalize() {

        shamlog_debug_mpi_ln(
            "PatchDataField MPI Comm",
            "finalizing mpi sycl comm ",
            comm_sz,
            int(comm_op),
            int(comm_mode));

        sycl_buf = std::make_unique<sycl::buffer<T>>(comm_sz);

        if (comm_mode == CopyToHost && comm_op == Send) {

            impl::copy_to_host::send::finalize<T>(comm_ptr);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            impl::copy_to_host::recv::finalize<T>(sycl_buf, comm_ptr, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            impl::directgpu::send::finalize<T>(comm_ptr);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            impl::directgpu::recv::finalize<T>(sycl_buf, comm_ptr, comm_sz);

        } else {
            logger::err_ln(
                "PatchDataField MPI Comm",
                "communication mode & op combination not implemented :",
                int(comm_mode),
                int(comm_op));
        }
    }

#define X(a) template struct BufferMpiRequest<a>;
    XMAC_SYCLMPI_TYPE_ENABLED
#undef X

} // namespace mpi_sycl_interop
