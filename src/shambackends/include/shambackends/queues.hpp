// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file queues.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include "shambase/memory.hpp"

namespace sham::queues {
    enum QueueKind { Compute, Alternative };

    struct QueueDetails {
        std::unique_ptr<sycl::queue> queue;
        u32 queue_global_id;
        bool direct_mpi_comm_capable = false;

        inline sycl::queue &get_queue() { return shambase::get_check_ref(queue); }
    };


} // namespace sham::queues
namespace sham {

    namespace backend {

        struct InitConfig {
            struct ById {
                u32 alt_queue_id;
                u32 compute_queue_id;
            };
            struct ByKey {
                std::string platform_search_key;
            };
        };

        void init(InitConfig cfg);

        void init_manual(
            std::vector<queues::QueueDetails> &&compute,
            std::vector<queues::QueueDetails> &&alternative);

        bool is_initialized();

        void close();
    } // namespace backend

    queues::QueueDetails &get_queue_details(u32 id = 0, queues::QueueKind kind = queues::Compute);

    sycl::queue &get_queue(u32 id = 0, queues::QueueKind kind = queues::Compute);


    struct QueueId{
        u32 id;
        queues::QueueKind kind;

        inline sycl::queue & get_queue(){
            return ::sham::get_queue(id, kind);
        }
        inline queues::QueueDetails & get_queue_details(){
            return ::sham::get_queue_details(id, kind);
        } 
    };

    inline sycl::queue & get_queue(QueueId id){
        return id.get_queue();
    }
    inline queues::QueueDetails & get_queue_details(QueueId id){
        return id.get_queue_details();
    }

    inline QueueId get_queue_id(u32 id = 0, queues::QueueKind kind = queues::Compute){
        return {id, kind};
    }

} // namespace sham