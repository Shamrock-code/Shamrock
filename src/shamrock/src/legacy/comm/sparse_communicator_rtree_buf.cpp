// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_communicator_rtree_buf.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/comm/sparse_communicator.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamtree/RadixTree.hpp"

template<class T>
struct SparseCommExchanger<RadixTreeField<T>> {

    // TODO emit warning that here the sycl buffer will be used with it's internal size
    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    static SparseCommResult<RadixTreeField<T>> sp_xchg(
        SparsePatchCommunicator &communicator,
        const SparseCommSource<RadixTreeField<T>> &send_comm_pdat) {
        StackEntry stack_loc{};
        using namespace shamrock::patch;

        SparseCommResult<RadixTreeField<T>> recv_obj;

        if (!send_comm_pdat.empty()) {

            u32 nvar = send_comm_pdat[0]->nvar;

            std::vector<mpi_sycl_interop::BufferMpiRequest<T>> rq_lst;

            u64 dtcnt = 0;

            {
                for (u64 i = 0; i < communicator.send_comm_vec.size(); i++) {
                    const Patch &psend
                        = communicator.global_patch_list[communicator.send_comm_vec[i].x()];
                    const Patch &precv
                        = communicator.global_patch_list[communicator.send_comm_vec[i].y()];

                    // if(precv.node_owner_id >= shamcomm::world_size()){
                    //     throw "";
                    // }

                    auto &send_buf = send_comm_pdat[i]->radix_tree_field_buf;

                    if (psend.node_owner_id == precv.node_owner_id) {
                        auto &vec = recv_obj[precv.id_patch];
                        dtcnt += send_buf->byte_size();
                        vec.push_back(
                            {psend.id_patch,
                             std::make_unique<RadixTreeField<T>>(RadixTreeField<T>{
                                 nvar,
                                 shamalgs::memory::duplicate(
                                     shamsys::instance::get_compute_scheduler_ptr()->get_queue().q,
                                     send_buf)})});
                    } else {
                        // std::cout << "send : " << shamcomm::world_rank() << " " <<
                        // precv.node_owner_id << std::endl;
                        dtcnt += mpi_sycl_interop::isend(
                            send_buf,
                            send_buf->size(),
                            rq_lst,
                            precv.node_owner_id,
                            communicator.local_comm_tag[i],
                            MPI_COMM_WORLD);
                    }
                }
            }

            if (communicator.global_comm_vec.size() > 0) {

                // std::cout << std::endl;
                for (u64 i = 0; i < communicator.global_comm_vec.size(); i++) {

                    const Patch &psend
                        = communicator.global_patch_list[communicator.global_comm_vec[i].x()];
                    const Patch &precv
                        = communicator.global_patch_list[communicator.global_comm_vec[i].y()];

                    if (precv.node_owner_id == shamcomm::world_rank()) {

                        if (psend.node_owner_id != precv.node_owner_id) {

                            recv_obj[precv.id_patch].push_back(
                                {psend.id_patch,
                                 std::make_unique<
                                     RadixTreeField<T>>()}); // patchdata_irecv(recv_rq,
                                                             // psend.node_owner_id,
                                                             // global_comm_tag[i],
                                                             // MPI_COMM_WORLD)}

                            auto &ref_write = std::get<1>(
                                recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]);
                            mpi_sycl_interop::irecv_probe<T>(
                                ref_write->radix_tree_field_buf,
                                rq_lst,
                                psend.node_owner_id,
                                communicator.global_comm_tag[i],
                                MPI_COMM_WORLD);
                        }
                    }
                }
                // std::cout << std::endl;
            }

            mpi_sycl_interop::waitall(rq_lst);

            communicator.xcgh_byte_cnt += dtcnt;

            // TODO check that this sort is valid
            for (auto &[key, obj] : recv_obj) {
                std::sort(obj.begin(), obj.end(), [](const auto &lhs, const auto &rhs) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                });
            }
        }

        return std::move(recv_obj);
    }
};

#define X(_arg_) template struct SparseCommExchanger<RadixTreeField<_arg_>>;
XMAC_LIST_ENABLED_FIELD
#undef X
