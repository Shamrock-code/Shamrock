// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata_field.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

//%Impl status : Good

#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
// #include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_cp_to_host.hpp"
// #include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_directgpu.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <cstdio>
#include <memory>

// TODO use hash for name + nvar to check if the field match before doing operation on them

namespace patchdata_field {

    comm_type current_mode = CopyToHost;

    template<class T>
    PatchDataFieldMpiRequest<T>::PatchDataFieldMpiRequest(
        PatchDataField<T> &pdat_field, comm_type comm_mode, op_type comm_op, u32 comm_val_cnt)
        : comm_mode(comm_mode), comm_op(comm_op), comm_val_cnt(comm_val_cnt),
          pdat_field(pdat_field) {

        shamlog_debug_mpi_ln(
            "PatchDataField MPI Comm",
            "starting mpi sycl comm ",
            comm_val_cnt,
            int(comm_op),
            int(comm_mode));

        if (comm_mode == CopyToHost && comm_op == Send) {

            comm_ptr = impl::copy_to_host::send::init<T>(
                std::make_unique<sycl::buffer<T>>(pdat_field.get_buf().copy_to_sycl_buffer()),
                comm_val_cnt);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            comm_ptr = impl::copy_to_host::recv::init<T>(comm_val_cnt);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            comm_ptr = impl::directgpu::send::init<T>(
                std::make_unique<sycl::buffer<T>>(pdat_field.get_buf().copy_to_sycl_buffer()),
                comm_val_cnt);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            comm_ptr = impl::directgpu::recv::init<T>(comm_val_cnt);

        } else {
            logger::err_ln(
                "PatchDataField MPI Comm",
                "communication mode & op combination not implemented :",
                int(comm_mode),
                int(comm_op));
        }
    }

    template<class T>
    void PatchDataFieldMpiRequest<T>::finalize() {

        shamlog_debug_mpi_ln(
            "PatchDataField MPI Comm",
            "finalizing mpi sycl comm ",
            comm_val_cnt,
            int(comm_op),
            int(comm_mode));

        if (comm_op == Recv_Probe) {
            pdat_field.resize(comm_val_cnt / pdat_field.get_nvar());
        }

        if (comm_mode == CopyToHost && comm_op == Send) {

            impl::copy_to_host::send::finalize<T>(comm_ptr);

        } else if (comm_mode == CopyToHost && comm_op == Recv_Probe) {

            auto buf_recv
                = std::make_unique<sycl::buffer<T>>(pdat_field.get_buf().copy_to_sycl_buffer());

            impl::copy_to_host::recv::finalize<T>(buf_recv, comm_ptr, comm_val_cnt);

            pdat_field.get_buf().copy_from_sycl_buffer(*buf_recv);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            impl::directgpu::send::finalize<T>(comm_ptr);

        } else if (comm_mode == DirectGPU && comm_op == Recv_Probe) {

            auto buf_recv
                = std::make_unique<sycl::buffer<T>>(pdat_field.get_buf().copy_to_sycl_buffer());

            impl::copy_to_host::recv::finalize<T>(buf_recv, comm_ptr, comm_val_cnt);

            pdat_field.get_buf().copy_from_sycl_buffer(*buf_recv);

        } else {
            logger::err_ln(
                "PatchDataField MPI Comm",
                "communication mode & op combination not implemented :",
                int(comm_mode),
                int(comm_op));
        }
    }

#define X(a) template struct PatchDataFieldMpiRequest<a>;
    XMAC_LIST_ENABLED_FIELD
#undef X

} // namespace patchdata_field
