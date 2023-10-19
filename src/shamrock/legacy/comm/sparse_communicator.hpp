// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparse_communicator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "aliases.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamrock/patch/Patch.hpp"
#include <vector>

#include "shamrock/scheduler/scheduler_mpi.hpp"



template <class T> using SparseCommSource = std::vector<std::unique_ptr<T>>;
template <class T> using SparseCommResult = std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<T>>>>;



class SparsePatchCommunicator;

template <class T> 
struct SparseCommExchanger{
    static SparseCommResult<T> sp_xchg(SparsePatchCommunicator & communicator, const SparseCommSource<T> &send_comm_pdat);
};

class SparsePatchCommunicator {
    

    std::vector<i32> local_comm_tag;

    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;

    u64 xcgh_byte_cnt = 0;


  public:
    std::vector<shamrock::patch::Patch> & global_patch_list;
    std::vector<u64_2> send_comm_vec;

    SparsePatchCommunicator(std::vector<shamrock::patch::Patch> & global_patch_list, std::vector<u64_2> send_comm_vec)
        : global_patch_list(global_patch_list), send_comm_vec(std::move(send_comm_vec)),
          local_comm_tag(send_comm_vec.size()) {}

    inline void fetch_comm_table() {StackEntry stack_loc{};

        {
            i32 iterator = 0;
            for (u64 i = 0; i < send_comm_vec.size(); i++) {
                local_comm_tag[i] = iterator;
                iterator++;
            }
        }

        shamalgs::collective::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
        shamalgs::collective::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
        

        xcgh_byte_cnt += 
            (send_comm_vec.size() * sizeof(u64) * 2)  + 
            (global_comm_vec.size() * sizeof(u64) * 2)  + 
            (local_comm_tag.size() * sizeof(i32))  + 
            (global_comm_tag.size() * sizeof(i32));
    }

    [[nodiscard]] inline u64 get_xchg_byte_count() const {
        return xcgh_byte_cnt;
    }

    inline void reset_xchg_byte_count(){
        xcgh_byte_cnt = 0;
    }

    template<typename T>
    friend struct SparseCommExchanger;

    template<class T>
    inline SparseCommResult<T> sparse_exchange(const SparseCommSource<T> &send_comm_pdat){
        return SparseCommExchanger<T>::sp_xchg(*this, send_comm_pdat);
    }
};







#include "shamrock/legacy/patch/base/patchdata.hpp"

template <> 
struct SparseCommExchanger<shamrock::patch::PatchData>{
    static SparseCommResult<shamrock::patch::PatchData> sp_xchg(SparsePatchCommunicator & communicator, const SparseCommSource<shamrock::patch::PatchData> &send_comm_pdat){
StackEntry stack_loc{};
        using namespace shamrock::patch;

        SparseCommResult<PatchData> recv_obj;

        if(!send_comm_pdat.empty()){
        
            PatchDataLayout & pdl = send_comm_pdat[0]->pdl;

            std::vector<PatchDataMpiRequest> rq_lst;


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

                    if (precv.node_owner_id == shamsys::instance::world_rank) {

                        if (psend.node_owner_id != precv.node_owner_id) {
                            recv_obj[precv.id_patch].push_back({psend.id_patch, std::make_unique<PatchData>(pdl)}); 
                            patchdata_irecv_probe(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                            rq_lst, psend.node_owner_id, communicator.global_comm_tag[i], MPI_COMM_WORLD);
                        }
                    }
                }
            }

            waitall_pdat_mpi_rq(rq_lst);


            communicator.xcgh_byte_cnt += dtcnt;


            for(auto & [key,obj] : recv_obj){
                std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                });
            }

        }

        return recv_obj;
    }
};

