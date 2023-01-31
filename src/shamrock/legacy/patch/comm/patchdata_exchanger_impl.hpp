// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include "shamrock/legacy/patch/base/patch.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <vector>

#include "shamrock/legacy/io/logs.hpp"
#include "shamrock/tree/RadixTree.hpp"
namespace patchdata_exchanger {


namespace impl{



template<class T>
inline void vector_isend(std::vector<T> &p, std::vector<MPI_Request> &rq_lst, i32 rank_dest, i32 tag, MPI_Comm comm){
    rq_lst.resize(rq_lst.size() + 1);
    mpi::isend(p.data(), p.size(), get_mpi_type<T>(), rank_dest, tag, comm, &rq_lst[rq_lst.size() - 1]);
}

template<class T>
inline void vector_irecv(std::vector<T> & pdat, std::vector<MPI_Request> &rq_lst, i32 rank_source, i32 tag, MPI_Comm comm){
    MPI_Status st;
    i32 cnt;
    int i = mpi::probe(rank_source, tag,comm, & st);
    mpi::get_count(&st, get_mpi_type<T>(), &cnt);
    rq_lst.resize(rq_lst.size() + 1);
    pdat.resize(cnt);
    mpi::irecv(pdat.data(), cnt, get_mpi_type<T>(), rank_source, tag, comm, &rq_lst[rq_lst.size() - 1]);
}


inline void make_comm_table(
    const std::vector<Patch> & in_global_patch_list,
    const std::vector<u64_2> & in_send_comm_vec,
    std::vector<u64_2> & out_global_comm_vec,
    std::vector<i32> & out_global_comm_tag,
    std::vector<i32> & out_local_comm_tag){

    out_local_comm_tag.resize(in_send_comm_vec.size());
    {
        i32 iterator = 0;
        for (u64 i = 0; i < in_send_comm_vec.size(); i++) {
            const Patch &psend = in_global_patch_list[in_send_comm_vec[i].x()];
            const Patch &precv = in_global_patch_list[in_send_comm_vec[i].y()];

            out_local_comm_tag[i] = iterator;

            iterator++;
        }
    }

    auto timer_allgatherv = timings::start_timer("allgatherv", timings::mpi);
    mpi_handler::vector_allgatherv(in_send_comm_vec, mpi_type_u64_2, out_global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
    mpi_handler::vector_allgatherv(out_local_comm_tag, mpi_type_i32, out_global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
    timer_allgatherv.stop();
}







inline void patch_data_exchange_object(
    PatchDataLayout & pdl,
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchData>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> & recv_obj
    ){

    //TODO enable if ultra verbose
    // std::cout << "len comm_pdat : " << send_comm_pdat.size() << std::endl;
    // std::cout << "len comm_vec : " << send_comm_vec.size() << std::endl;

    auto timer_transf = timings::start_timer("patch_data_exchange_object", timings::function);

    std::vector<i32> local_comm_tag(send_comm_vec.size());
    {
        i32 iterator = 0;
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            local_comm_tag[i] = iterator;

            iterator++;
        }
    }

    auto timer_allgatherv = timings::start_timer("allgatherv", timings::mpi);
    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;
    mpi_handler::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
    mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
    timer_allgatherv.stop();

    std::vector<PatchDataMpiRequest> rq_lst;

    auto timer_transfmpi = timings::start_timer("patchdata_exchanger", timings::mpi);

    u64 dtcnt = 0;

    {
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            if (psend.node_owner_id == precv.node_owner_id) {
                // std::cout << "same node !!!\n";
                recv_obj[precv.id_patch].push_back({psend.id_patch, std::move(send_comm_pdat[i])});
                send_comm_pdat[i] = nullptr;
            } else {
                //TODO enable if ultra verbose
                //std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                //                    psend.node_owner_id, precv.node_owner_id, local_comm_tag[i]);
                dtcnt += patchdata_isend(*send_comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
            }

            // std::cout << format("send : (%3d,%3d) : %d -> %d /
            // %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
            // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
        }
    }

    if (global_comm_vec.size() > 0) {

        //std::cout << std::endl;
        for (u64 i = 0; i < global_comm_vec.size(); i++) {

            const Patch &psend = global_patch_list[global_comm_vec[i].x()];
            const Patch &precv = global_patch_list[global_comm_vec[i].y()];
            // std::cout << format("(%3d,%3d) : %d -> %d /
            // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

            if (precv.node_owner_id == shamsys::instance::world_rank) {

                if (psend.node_owner_id != precv.node_owner_id) {
                    //TODO enable if ultra verbose
                    // std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                    //                     psend.node_owner_id, precv.node_owner_id, global_comm_tag[i]);
                    recv_obj[precv.id_patch].push_back(
                        {psend.id_patch, std::make_unique<PatchData>(pdl)}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                                                          // global_comm_tag[i], MPI_COMM_WORLD)}
                    dtcnt += patchdata_irecv_probe(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                    rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }

                // std::cout << format("recv (%3d,%3d) : %d -> %d /
                // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                // Interface_map[precv.id_patch].push_back({psend.id_patch, new PatchData()});//patchdata_irecv(recv_rq,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
            }
        }
        //std::cout << std::endl;
    }

    waitall_pdat_mpi_rq(rq_lst);

    timer_transfmpi.stop(dtcnt);

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */
    
    //TODO check that this sort is valid
    for(auto & [key,obj] : recv_obj){
        std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
    }

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */

    timer_transf.stop();

}






template<class T>
inline void patch_data_field_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchDataField<T>>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>> & recv_obj
    ){

    //TODO enable if ultra verbose
    // std::cout << "len comm_pdat : " << send_comm_pdat.size() << std::endl;
    // std::cout << "len comm_vec : " << send_comm_vec.size() << std::endl;

    auto timer_transf = timings::start_timer("patch_data_field_exchange_object", timings::function);


    std::vector<i32> local_comm_tag(send_comm_vec.size());
    {
        i32 iterator = 0;
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            local_comm_tag[i] = iterator;

            iterator++;
        }
    }

    auto timer_allgatherv = timings::start_timer("allgatherv", timings::mpi);
    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;
    mpi_handler::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
    mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
    timer_allgatherv.stop();


    std::vector<patchdata_field::PatchDataFieldMpiRequest<T>> rq_lst;

    auto timer_transfmpi = timings::start_timer("patchdata_exchanger", timings::mpi);

    u64 dtcnt = 0;

    {
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            if (psend.node_owner_id == precv.node_owner_id) {
                // std::cout << "same node !!!\n";
                recv_obj[precv.id_patch].push_back({psend.id_patch, std::move(send_comm_pdat[i])});
                send_comm_pdat[i] = nullptr;
            } else {
                //TODO enable if ultra verbose
                //std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                //                    psend.node_owner_id, precv.node_owner_id, local_comm_tag[i]);
                dtcnt += patchdata_field::isend<T>(*send_comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
            }

            // std::cout << format("send : (%3d,%3d) : %d -> %d /
            // %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
            // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
        }
    }

    if (global_comm_vec.size() > 0) {

        //std::cout << std::endl;
        for (u64 i = 0; i < global_comm_vec.size(); i++) {

            const Patch &psend = global_patch_list[global_comm_vec[i].x()];
            const Patch &precv = global_patch_list[global_comm_vec[i].y()];
            // std::cout << format("(%3d,%3d) : %d -> %d /
            // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

            if (precv.node_owner_id == shamsys::instance::world_rank) {

                if (psend.node_owner_id != precv.node_owner_id) {
                    //TODO enable if ultra verbose
                    // std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                    //                     psend.node_owner_id, precv.node_owner_id, global_comm_tag[i]);
                    recv_obj[precv.id_patch].push_back(
                        {psend.id_patch, std::make_unique<PatchDataField<T>>("comp_field",1)}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                                                          // global_comm_tag[i], MPI_COMM_WORLD)}
                    dtcnt += patchdata_field::irecv_probe<T>(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                    rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }

                // std::cout << format("recv (%3d,%3d) : %d -> %d /
                // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                // Interface_map[precv.id_patch].push_back({psend.id_patch, new PatchData()});//patchdata_irecv(recv_rq,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
            }
        }
        //std::cout << std::endl;
    }

    patchdata_field::waitall(rq_lst);

    timer_transfmpi.stop(dtcnt);

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */

    //TODO check that this sort is valid
    for(auto & [key,obj] : recv_obj){
        std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
    }

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */

    timer_transf.stop();

}






template<class u_morton, class vec3>
inline void radix_tree_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<RadixTree<u_morton, vec3,3>>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<RadixTree<u_morton, vec3,3>>>>> & recv_obj
    ){

    //TODO enable if ultra verbose
    // std::cout << "len comm_pdat : " << send_comm_pdat.size() << std::endl;
    // std::cout << "len comm_vec : " << send_comm_vec.size() << std::endl;

    auto timer_transf = timings::start_timer("patch_data_field_exchange_object", timings::function);


    std::vector<i32> local_comm_tag(send_comm_vec.size());
    {
        i32 iterator = 0;
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            local_comm_tag[i] = iterator;

            iterator++;
        }
    }

    auto timer_allgatherv = timings::start_timer("allgatherv", timings::mpi);
    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;
    mpi_handler::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
    mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
    timer_allgatherv.stop();


    std::vector<tree_comm::RadixTreeMPIRequest<u_morton, vec3>> rq_lst;

    auto timer_transfmpi = timings::start_timer("radix_tree_exchanger", timings::mpi);

    {
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            if (psend.node_owner_id == precv.node_owner_id) {
                // std::cout << "same node !!!\n";
                recv_obj[precv.id_patch].push_back({psend.id_patch, std::move(send_comm_pdat[i])});
                send_comm_pdat[i] = nullptr;
            } else {
                //TODO enable if ultra verbose
                //std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                //                    psend.node_owner_id, precv.node_owner_id, local_comm_tag[i]);
                tree_comm::comm_isend(*send_comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
            }

            // std::cout << format("send : (%3d,%3d) : %d -> %d /
            // %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
            // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
        }
    }

    if (global_comm_vec.size() > 0) {

        //std::cout << std::endl;
        for (u64 i = 0; i < global_comm_vec.size(); i++) {

            const Patch &psend = global_patch_list[global_comm_vec[i].x()];
            const Patch &precv = global_patch_list[global_comm_vec[i].y()];
            // std::cout << format("(%3d,%3d) : %d -> %d /
            // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

            if (precv.node_owner_id == shamsys::instance::world_rank) {

                if (psend.node_owner_id != precv.node_owner_id) {
                    //TODO enable if ultra verbose
                    // std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                    //                     psend.node_owner_id, precv.node_owner_id, global_comm_tag[i]);
                    recv_obj[precv.id_patch].push_back(
                        {psend.id_patch, std::make_unique<RadixTree<u_morton, vec3,3>>(RadixTree<u_morton, vec3,3>::make_empty())}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                                                          // global_comm_tag[i], MPI_COMM_WORLD)}
                    tree_comm::comm_irecv_probe(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                    rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }

                // std::cout << format("recv (%3d,%3d) : %d -> %d /
                // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                // Interface_map[precv.id_patch].push_back({psend.id_patch, new PatchData()});//patchdata_irecv(recv_rq,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
            }
        }
        //std::cout << std::endl;
    }

    tree_comm::wait_all(rq_lst);

    timer_transfmpi.stop();

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */

    //TODO check that this sort is valid
    for(auto & [key,obj] : recv_obj){
        std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
    }

    /*
    for(auto & [id_ps, pdat] : (recv_obj[0])){
        std::cout << "int : " << 0 << " <- " << id_ps << " : " << pdat->size() << std::endl;
    }std::cout << std::endl;
    */

    timer_transf.stop();

}






} // namespace impl





    



} // namespace patchdata_exchanger