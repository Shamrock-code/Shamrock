// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_communicator_radix_tree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/comm/sparse_communicator.hpp"
#include "shamtree/RadixTree.hpp"

template<class u_morton, class vec3>
struct SparseCommExchanger<RadixTree<u_morton, vec3>> {

    [[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
    static SparseCommResult<RadixTree<u_morton, vec3>> sp_xchg(
        SparsePatchCommunicator &communicator,
        const SparseCommSource<RadixTree<u_morton, vec3>> &send_comm_pdat) {
        StackEntry stack_loc{};
        using namespace shamrock::patch;

        SparseCommResult<RadixTree<u_morton, vec3>> recv_obj;

        if (!send_comm_pdat.empty()) {

            std::vector<tree_comm::RadixTreeMPIRequest<u_morton, vec3>> rq_lst;

            u64 dtcnt = 0;

            {
                for (u64 i = 0; i < communicator.send_comm_vec.size(); i++) {
                    const Patch &psend
                        = communicator.global_patch_list[communicator.send_comm_vec[i].x()];
                    const Patch &precv
                        = communicator.global_patch_list[communicator.send_comm_vec[i].y()];

                    if (psend.node_owner_id == precv.node_owner_id) {
                        auto &vec = recv_obj[precv.id_patch];
                        dtcnt += send_comm_pdat[i]->memsize();
                        vec.push_back({psend.id_patch, send_comm_pdat[i]->duplicate_to_ptr()});
                    } else {

                        dtcnt += tree_comm::comm_isend(
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
                                 std::make_unique<RadixTree<u_morton, vec3>>(
                                     RadixTree<u_morton, vec3>::
                                         make_empty())}); // patchdata_irecv(recv_rq,
                                                          // psend.node_owner_id,
                                                          // global_comm_tag[i], MPI_COMM_WORLD)}
                            tree_comm::comm_irecv_probe(
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

            tree_comm::wait_all(rq_lst);

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

template struct SparseCommExchanger<RadixTree<u32, f32_3>>;
template struct SparseCommExchanger<RadixTree<u64, f32_3>>;
template struct SparseCommExchanger<RadixTree<u32, f64_3>>;
template struct SparseCommExchanger<RadixTree<u64, f64_3>>;
