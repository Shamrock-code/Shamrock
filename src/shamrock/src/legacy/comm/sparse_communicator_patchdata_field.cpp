// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_communicator_patchdata_field.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/comm/sparse_communicator.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"

template<class T>
struct SparseCommExchanger<PatchDataField<T>> {

    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    static SparseCommResult<PatchDataField<T>> sp_xchg(
        SparsePatchCommunicator &communicator,
        const SparseCommSource<PatchDataField<T>> &send_comm_pdat) {
        StackEntry stack_loc{};
        using namespace shamrock::patch;

        SparseCommResult<PatchDataField<T>> recv_obj;

        if (!send_comm_pdat.empty()) {

            std::vector<patchdata_field::PatchDataFieldMpiRequest<T>> rq_lst;

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

                    if (psend.node_owner_id == precv.node_owner_id) {
                        auto &vec = recv_obj[precv.id_patch];
                        dtcnt += send_comm_pdat[i]->memsize();
                        vec.push_back({psend.id_patch, send_comm_pdat[i]->duplicate_to_ptr()});
                    } else {
                        // std::cout << "send : " << shamcomm::world_rank() << " " <<
                        // precv.node_owner_id << std::endl;
                        dtcnt += patchdata_field::isend<T>(
                            *send_comm_pdat[i],
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
                                 std::make_unique<PatchDataField<T>>(
                                     "comp_field",
                                     1)}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                           // global_comm_tag[i], MPI_COMM_WORLD)}
                            patchdata_field::irecv_probe<T>(
                                *std::get<1>(
                                    recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                rq_lst,
                                psend.node_owner_id,
                                communicator.global_comm_tag[i],
                                MPI_COMM_WORLD);
                        }
                    }
                }
                // std::cout << std::endl;
            }

            patchdata_field::waitall(rq_lst);

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

#define X(_arg_) template struct SparseCommExchanger<PatchDataField<_arg_>>;
XMAC_LIST_ENABLED_FIELD
#undef X
