// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CommunicationBufferImpl.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * \todo this file should pull queues from backends and not sys lib
 */

#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shambackends/queues.hpp"
#include "shambase/exception.hpp"
#include <cstring>
#include <stdexcept>

namespace shamcomm::details {



    void CommunicationBuffer<CopyToHost>::alloc_usm(u64 len){
        usm_ptr = sycl::malloc_host<u8>(len,bind_queue);
    }

    void CommunicationBuffer<CopyToHost>::copy_to_usm(sycl::buffer<u8> & obj_ref, u64 len){   
        sycl::host_accessor acc {obj_ref, sycl::read_only};
        const u8* tmp = acc.get_pointer();
        bind_queue.memcpy(usm_ptr, tmp, len).wait();
    }

    sycl::buffer<u8> CommunicationBuffer<CopyToHost>::build_from_usm(u64 len){
        sycl::buffer<u8> buf_ret (len);
        {
            sycl::host_accessor acc {buf_ret, sycl::write_only, sycl::no_init};
            u8* tmp = acc.get_pointer();
            bind_queue.memcpy(tmp, usm_ptr, len).wait();
        }
        return buf_ret;
    }

    void CommunicationBuffer<CopyToHost>::copy_usm(u64 len, u8* new_usm){
        bind_queue.memcpy(new_usm, usm_ptr, len).wait();
    }



    ///////



    
    void CommunicationBuffer<DirectGPU>::alloc_usm(u64 len){
        usm_ptr = sycl::malloc_device<u8>(len,bind_queue);
    }

    
    void CommunicationBuffer<DirectGPU>::copy_to_usm(sycl::buffer<u8> & obj_ref, u64 len){
        
        auto ev = bind_queue.submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {obj_ref, cgh, sycl::read_only};

            u8* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr[i] = acc_buf[i];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    
    }

    
    sycl::buffer<u8> CommunicationBuffer<DirectGPU>::build_from_usm(u64 len){

        sycl::buffer<u8> buf_ret (len);


        auto ev = bind_queue.submit([&](sycl::handler & cgh){
            
            sycl::accessor acc_buf {buf_ret, cgh,sycl::write_only, sycl::no_init};

            u8* ptr = usm_ptr;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                acc_buf[i] = ptr[i];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls

        return buf_ret;
    }

    
    void CommunicationBuffer<DirectGPU>::copy_usm(u64 len, u8* new_usm){

        auto ev = bind_queue.submit([&](sycl::handler & cgh){
            
            u8* ptr = usm_ptr;

            u8* ptr_new = new_usm;

            cgh.parallel_for(sycl::range<1>{len},[=](sycl::item<1> i){
                ptr_new[i] = ptr[i];
            });

        });

        ev.wait();//TODO wait for the event only when doing MPI calls
    }

}