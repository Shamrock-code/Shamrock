// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CommunicationBuffer.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Shamrock communication buffers
 *
 */

#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"

namespace shamcomm {

    bool validate_comm_internal(std::shared_ptr<sham::DeviceScheduler> &device_sched) {

        u32 nbytes = 1e5;
        sycl::buffer<u8> buf_comp(nbytes);

        {
            sycl::host_accessor acc1{buf_comp, sycl::write_only, sycl::no_init};
            for (u32 i = 0; i < nbytes; i++) {
                acc1[i] = i % 100;
            }
        }

        shamcomm::CommunicationBuffer cbuf{buf_comp, device_sched};
        shamcomm::CommunicationBuffer cbuf_recv{nbytes, device_sched};

        MPI_Request rq1, rq2;
        if (shamcomm::world_rank() == shamcomm::world_size() - 1) {
            shamcomm::mpi::Isend(cbuf.get_ptr(), nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &rq1);
        }

        if (shamcomm::world_rank() == 0) {
            shamcomm::mpi::Irecv(
                cbuf_recv.get_ptr(),
                nbytes,
                MPI_BYTE,
                shamcomm::world_size() - 1,
                0,
                MPI_COMM_WORLD,
                &rq2);
        }

        if (shamcomm::world_rank() == shamcomm::world_size() - 1) {
            shamcomm::mpi::Wait(&rq1, MPI_STATUS_IGNORE);
        }

        if (shamcomm::world_rank() == 0) {
            shamcomm::mpi::Wait(&rq2, MPI_STATUS_IGNORE);
        }

        sycl::buffer<u8> recv = shamcomm::CommunicationBuffer::convert(std::move(cbuf_recv));

        bool valid = true;

        if (shamcomm::world_rank() == 0) {
            sycl::host_accessor acc1{buf_comp};
            sycl::host_accessor acc2{recv};

            std::string id_err_list = "errors in id : ";

            bool eq = true;
            for (u32 i = 0; i < recv.size(); i++) {
                if (!sham::equals(acc1[i], acc2[i])) {
                    eq = false;
                    // id_err_list += std::to_string(i) + " ";
                }
            }

            valid = eq;
        }

        return valid;
    }

    void validate_comm(std::shared_ptr<sham::DeviceScheduler> &sched) {
        u32 nbytes = 1e5;
        sycl::buffer<u8> buf_comp(nbytes);

        bool call_abort = false;

        bool dgpu_mode = sched->ctx->device->mpi_prop.is_mpi_direct_capable;

        using namespace shambase::term_colors;
        if (dgpu_mode) {
            if (validate_comm_internal(sched)) {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(" - MPI use Direct Comm :", col8b_green() + "Working" + reset());
            } else {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(" - MPI use Direct Comm :", col8b_red() + "Fail" + reset());
                if (shamcomm::world_rank() == 0)
                    logger::err_ln("Sys", "the select comm mode failed, try forcing dgpu mode off");
                call_abort = true;
            }
        } else {
            if (validate_comm_internal(sched)) {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(
                        " - MPI use Copy to Host :", col8b_green() + "Working" + reset());
            } else {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(" - MPI use Copy to Host :", col8b_red() + "Fail" + reset());
                call_abort = true;
            }
        }

        shamcomm::mpi::Barrier(MPI_COMM_WORLD);

        if (call_abort) {
            MPI_Abort(MPI_COMM_WORLD, 26);
        }
    }

} // namespace shamcomm
