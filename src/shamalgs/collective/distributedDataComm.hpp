// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file distributedDataComm.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/sparseXchg.hpp"
#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include <functional>
#include <mpi.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace shamalgs::collective {


    using SerializedDDataComm = shambase::DistributedDataShared<std::unique_ptr<sycl::buffer<u8>>>;

    void distributed_data_sparse_comm(SerializedDDataComm &send_ddistrib_data,
                                             SerializedDDataComm &recv_distrib_data,
                                             std::function<i32(u64)> rank_getter,
                                             std::optional<SparseCommTable> comm_table = {});

    template<class T>
    inline void serialize_sparse_comm(
        shambase::DistributedDataShared<T> && send_distrib_data,
        shambase::DistributedDataShared<T> &recv_distrib_data,
        std::function<i32(u64)> rank_getter,
        std::function<std::unique_ptr<sycl::buffer<u8>>(T &)> serialize,
        std::function<T(std::unique_ptr<sycl::buffer<u8>> &&)> deserialize,
        std::optional<SparseCommTable> comm_table = {}
        ){

        StackEntry stack_loc{};
            
        shambase::DistributedDataShared<T> same_rank_tmp;
        // allow move op for same rank
        send_distrib_data.tranfer_all([&](u64 l, u64 r){
            return rank_getter(l) == rank_getter(r);
        }, same_rank_tmp);

        SerializedDDataComm dcomm_send = 
            send_distrib_data.template map<std::unique_ptr<sycl::buffer<u8>>>(
                [&](u64 ,u64,T & obj){
                    return serialize(obj);
                }
            );

        SerializedDDataComm dcomm_recv;

        distributed_data_sparse_comm(
            dcomm_send, 
            dcomm_recv, 
            rank_getter);

        recv_distrib_data = 
            dcomm_recv.map<T>(
                [&](u64, u64, std::unique_ptr<sycl::buffer<u8>> & buf){
                    //exchange the buffer held by the distrib data and give it to the deserializer
                    return deserialize(std::exchange(buf, std::unique_ptr<sycl::buffer<u8>>{}));
                }
            );

        logger::debug_ln("SparseComm","skipped",same_rank_tmp.get_native().size(),"communications");

        same_rank_tmp.tranfer_all([&](u64 l, u64 r){
            return true;
        }, recv_distrib_data);

    }

    /**
     * @brief global ids = allgatherv(local_ids)
     * 
     * @tparam T 
     * @param src 
     * @param local_ids 
     * @param global_ids 
     * @return shambase::DistributedData<T> 
     */
    template<class T, class P>
    shambase::DistributedData<T> fetch_all_simple(shambase::DistributedData<T> & src, std::vector<P> local_ids, std::vector<P> global_ids, std::function<u64(P)> id_getter){
        std::vector<T> vec_local(local_ids.size());
        for(u32 i = 0; i < local_ids.size(); i++){
            vec_local[i] = src.get(id_getter(local_ids[i]));
        }

        std::vector<T> vec_global;
        vector_allgatherv(vec_local, get_mpi_type<T>(), vec_global, get_mpi_type<T>(), MPI_COMM_WORLD);

        shambase::DistributedData<T> ret;
        for(u32 i = 0; i < global_ids.size(); i++){
            ret.add_obj(id_getter(global_ids[i]), T(vec_global[i]));
        }
        return ret;
    }

    /**
     * @brief global ids = allgatherv(local_ids)
     * 
     * @tparam T 
     * @param src 
     * @param local_ids 
     * @param global_ids 
     * @return shambase::DistributedData<T> 
     */
    template<class T, class P>
    shambase::DistributedData<T> fetch_all_storeload(shambase::DistributedData<T> & src, std::vector<P> local_ids, std::vector<P> global_ids, std::function<u64(P)> id_getter){
        
        using Trepr = typename T::Tload_store_repr;
        constexpr u32 reprsz = T::sz_load_store_repr;
        
        std::vector<T> vec_local(local_ids.size()*reprsz);
        for(u32 i = 0; i < local_ids.size(); i++){
            src.get(id_getter(local_ids[i])).store(i*reprsz, vec_local);
        }

        std::vector<T> vec_global;
        vector_allgatherv(vec_local, get_mpi_type<T>(), vec_global, get_mpi_type<T>(), MPI_COMM_WORLD);

        shambase::DistributedData<T> ret;
        for(u32 i = 0; i < global_ids.size(); i++){
            T tmp = T::load(i*reprsz, vec_global);
            ret.add_obj(id_getter(global_ids[i]), std::move(tmp));
        }
        return ret;
    }



} // namespace shamalgs::collective