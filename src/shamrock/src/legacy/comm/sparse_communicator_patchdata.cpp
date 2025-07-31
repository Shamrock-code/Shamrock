// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_communicator_patchdata.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/comm/sparse_communicator.hpp"

#if false
    #include "shamrock/legacy/patch/base/patchdata.hpp"

template <>
struct SparseCommExchanger<PatchData>{
    static SparseCommResult<PatchData> sp_xchg(SparsePatchCommunicator & communicator, const SparseCommSource<PatchData> &send_comm_pdat){



        SparseCommResult<PatchData> recv_obj;

        if(!send_comm_pdat.empty()){

            PatchDataLayout & pdl = send_comm_pdat[0]->pdl;

            std::vector<PatchDataMpiRequest> rq_lst;

            auto timer_transfmpi = timings::start_timer("patchdata_exchanger", timings::mpi);

            u64 dtcnt = 0;

            {
                for (u64 i = 0; i < communicator.send_comm_vec.size(); i++) {
                    const Patch &psend = communicator.global_patch_list[communicator.send_comm_vec[i].x()];
                    const Patch &precv = communicator.global_patch_list[communicator.send_comm_vec[i].y()];

                    if (psend.node_owner_id == precv.node_owner_id) {
                        auto & vec = recv_obj[precv.id_patch];
                        dtcnt += send_comm_pdat[i]->memsize();
                        vec.push_back({psend.id_patch, send_comm_pdat[i]->duplicate_to_ptr()});
                    } else {

                        dtcnt += patchdata_isend(*send_comm_pdat[i], rq_lst, precv.node_owner_id, communicator.local_comm_tag[i], MPI_COMM_WORLD);
                    }

                }
            }

            if (communicator.global_comm_vec.size() > 0) {

                for (u64 i = 0; i < communicator.global_comm_vec.size(); i++) {

                    const Patch &psend = communicator.global_patch_list[communicator.global_comm_vec[i].x()];
                    const Patch &precv = communicator.global_patch_list[communicator.global_comm_vec[i].y()];

                    if (precv.node_owner_id == shamcomm::world_rank()) {

                        if (psend.node_owner_id != precv.node_owner_id) {
                            recv_obj[precv.id_patch].push_back({psend.id_patch, std::make_unique<PatchData>(pdl)});
                            patchdata_irecv_probe(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                            rq_lst, psend.node_owner_id, communicator.global_comm_tag[i], MPI_COMM_WORLD);
                        }
                    }
                }
            }

            waitall_pdat_mpi_rq(rq_lst);

            timer_transfmpi.stop(dtcnt);

            communicator.xcgh_byte_cnt += dtcnt;


            for(auto & [key,obj] : recv_obj){
                std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                });
            }

        }

        return std::move(recv_obj);
    }
};

#endif
