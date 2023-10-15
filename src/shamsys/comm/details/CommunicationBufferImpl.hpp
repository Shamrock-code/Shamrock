// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CommunicationBufferImpl.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shambase/exception.hpp"
#include "shambase/type_aliases.hpp"
#include "shambase/sycl.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>
#include <variant>

namespace shamsys {

    enum CommunicationProtocol{
        /**
         * @brief copy data to the host and then perform the call
         */
        CopyToHost, 
        
        /**
         * @brief copy data straight from the GPU
         */
        DirectGPU, 
        
    };

    inline CommunicationProtocol get_protocol(){
        if(shamsys::instance::is_direct_gpu_selected()){
            return DirectGPU;
        }else{
            return CopyToHost;
        }
    }

    namespace details {

        template<CommunicationProtocol comm_mode> 
        class CommunicationBuffer;

        template<>
        class CommunicationBuffer<CopyToHost> {

            u8* usm_ptr;
            u64 bytelen;

            void alloc_usm(u64 len);
            void copy_to_usm(sycl::buffer<u8> & obj_ref, u64 len);
            sycl::buffer<u8> build_from_usm(u64 len);

            void copy_usm(u64 len, u8* new_usm);

            public:
            inline CommunicationBuffer(u64 bytelen){
                if (bytelen == 0) {
                    throw shambase::throw_with_loc<std::invalid_argument>("can not create a buffer of size = 0");
                }
                this->bytelen = bytelen;
                alloc_usm(bytelen);
            }

            inline CommunicationBuffer(sycl::buffer<u8> & obj_ref){
                bytelen = obj_ref.size();
                alloc_usm(bytelen);
                copy_to_usm(obj_ref,bytelen);
            }

            inline CommunicationBuffer(sycl::buffer<u8> && moved_obj){
                bytelen = moved_obj.size();
                alloc_usm(bytelen);
                copy_to_usm(moved_obj,bytelen);
            }

            inline ~CommunicationBuffer(){
                sycl::free(usm_ptr,instance::get_compute_queue());
            }

            inline CommunicationBuffer(CommunicationBuffer&& other) noexcept : 
                usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
                bytelen(other.bytelen){

            } // move constructor

            inline CommunicationBuffer& operator=(CommunicationBuffer&& other) noexcept{
                std::swap(usm_ptr, other.usm_ptr);
                bytelen = (other.bytelen);

                return *this;
            } // move assignment

            inline std::unique_ptr<CommunicationBuffer> duplicate_to_ptr(){
                std::unique_ptr<CommunicationBuffer> ret = std::make_unique<CommunicationBuffer>(bytelen);
                copy_usm(bytelen, ret->usm_ptr);
                return ret;
            }


            inline sycl::buffer<u8> copy_back(){
                u64 len = bytelen;

                return build_from_usm(len);
            }

            inline u64 get_bytesize(){
                return bytelen;
            }

            static sycl::buffer<u8> convert(CommunicationBuffer && buf){
                return buf.copy_back();
            }

            u8* get_ptr(){
                return usm_ptr;
            }

        };

        
        template<>
        class CommunicationBuffer<DirectGPU> {

            u8* usm_ptr;
            u64 bytelen;

            void alloc_usm(u64 len);
            void copy_to_usm(sycl::buffer<u8> & obj_ref, u64 len);
            sycl::buffer<u8> build_from_usm(u64 len);

            void copy_usm(u64 len, u8* new_usm);

            public:
            inline CommunicationBuffer(u64 bytelen){
                if (bytelen == 0) {
                    throw shambase::throw_with_loc<std::invalid_argument>("can not create a buffer of size = 0");
                }
                this->bytelen = bytelen;
                alloc_usm(bytelen);
            }

            inline CommunicationBuffer(sycl::buffer<u8> & obj_ref){
                bytelen = obj_ref.size();
                alloc_usm(bytelen);
                copy_to_usm(obj_ref,bytelen);
            }

            inline CommunicationBuffer(sycl::buffer<u8> && moved_obj){
                bytelen = moved_obj.size();
                alloc_usm(bytelen);
                copy_to_usm(moved_obj,bytelen);
            }

            inline ~CommunicationBuffer(){
                sycl::free(usm_ptr,instance::get_compute_queue());
            }

            inline CommunicationBuffer(CommunicationBuffer&& other) noexcept : 
                usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
                bytelen(other.bytelen){

            } // move constructor

            inline CommunicationBuffer& operator=(CommunicationBuffer&& other) noexcept{
                std::swap(usm_ptr, other.usm_ptr);
                bytelen = (other.bytelen);

                return *this;
            } // move assignment

            inline std::unique_ptr<CommunicationBuffer> duplicate_to_ptr(){
                std::unique_ptr<CommunicationBuffer> ret = std::make_unique<CommunicationBuffer>(bytelen);
                copy_usm(bytelen, ret->usm_ptr);
                return ret;
            }


            inline sycl::buffer<u8> copy_back(){
                u64 len = bytelen;

                return build_from_usm(len);
            }

            inline u64 get_bytesize(){
                return bytelen;
            }

            static sycl::buffer<u8> convert(CommunicationBuffer && buf){
                return buf.copy_back();
            }

            u8* get_ptr(){
                return usm_ptr;
            }

        };
    } // namespace details

}