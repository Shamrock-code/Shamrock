/**
 * @file patchdata_buffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "patch/patchdata.hpp"
#include "patch/patchdata_field.hpp"
#include "patch/patchdata_layout.hpp"
#include "sys/sycl_handler.hpp"

/**
 * @brief sycl buffer loaded version of PatchData
 * 
 */
class PatchDataBuffer{ public:

    u32 element_count;
    template<class T>
    struct PatchDataFieldBuffer{
        std::unique_ptr<sycl::buffer<T>> buf;
        PatchDataField<T> & field_ref;
    };




    PatchDataLayout & pdl;

    std::vector<PatchDataFieldBuffer<f32   >> fields_f32;
    std::vector<PatchDataFieldBuffer<f32_2 >> fields_f32_2;
    std::vector<PatchDataFieldBuffer<f32_3 >> fields_f32_3;
    std::vector<PatchDataFieldBuffer<f32_4 >> fields_f32_4;
    std::vector<PatchDataFieldBuffer<f32_8 >> fields_f32_8;
    std::vector<PatchDataFieldBuffer<f32_16>> fields_f32_16;

    std::vector<PatchDataFieldBuffer<f64   >> fields_f64;
    std::vector<PatchDataFieldBuffer<f64_2 >> fields_f64_2;
    std::vector<PatchDataFieldBuffer<f64_3 >> fields_f64_3;
    std::vector<PatchDataFieldBuffer<f64_4 >> fields_f64_4;
    std::vector<PatchDataFieldBuffer<f64_8 >> fields_f64_8;
    std::vector<PatchDataFieldBuffer<f64_16>> fields_f64_16;

    std::vector<PatchDataFieldBuffer<u32   >> fields_u32;

    std::vector<PatchDataFieldBuffer<u64   >> fields_u64;

    inline PatchDataBuffer(PatchDataLayout & pdl) : pdl(pdl) {}
};



inline PatchDataBuffer attach_to_patchData(PatchData & pdat){
    PatchDataBuffer pdatbuf(pdat.patchdata_layout);
    
    pdatbuf.element_count = u32(pdat.get_obj_cnt());


    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32.size(); idx++){
        std::unique_ptr<sycl::buffer<f32>> buf;

        if(pdat.fields_f32[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32>>(pdat.fields_f32[idx].data(),pdat.fields_f32[idx].size());
        }

        pdatbuf.fields_f32.push_back({std::move(buf),pdat.fields_f32[idx]});
    }

    
    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_2.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_2>> buf;

        if(pdat.fields_f32_2[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_2>>(pdat.fields_f32_2[idx].data(),pdat.fields_f32_2[idx].size());
        }

        pdatbuf.fields_f32_2.push_back({std::move(buf),pdat.fields_f32_2[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_3.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_3>> buf;

        if(pdat.fields_f32_3[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_3>>(pdat.fields_f32_3[idx].data(),pdat.fields_f32_3[idx].size());
        }

        pdatbuf.fields_f32_3.push_back({std::move(buf),pdat.fields_f32_3[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_4.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_4>> buf;

        if(pdat.fields_f32_4[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_4>>(pdat.fields_f32_4[idx].data(),pdat.fields_f32_4[idx].size());
        }

        pdatbuf.fields_f32_4.push_back({std::move(buf),pdat.fields_f32_4[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_8.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_8>> buf;

        if(pdat.fields_f32_8[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_8>>(pdat.fields_f32_8[idx].data(),pdat.fields_f32_8[idx].size());
        }

        pdatbuf.fields_f32_8.push_back({std::move(buf),pdat.fields_f32_8[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_16.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_16>> buf;

        if(pdat.fields_f32_16[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_16>>(pdat.fields_f32_16[idx].data(),pdat.fields_f32_16[idx].size());
        }

        pdatbuf.fields_f32_16.push_back({std::move(buf),pdat.fields_f32_16[idx]});
    }
    




    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64.size(); idx++){
        std::unique_ptr<sycl::buffer<f64>> buf;

        if(pdat.fields_f64[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64>>(pdat.fields_f64[idx].data(),pdat.fields_f64[idx].size());
        }

        pdatbuf.fields_f64.push_back({std::move(buf),pdat.fields_f64[idx]});
    }

    
    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_2.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_2>> buf;

        if(pdat.fields_f64_2[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_2>>(pdat.fields_f64_2[idx].data(),pdat.fields_f64_2[idx].size());
        }

        pdatbuf.fields_f64_2.push_back({std::move(buf),pdat.fields_f64_2[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_3.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_3>> buf;

        if(pdat.fields_f64_3[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_3>>(pdat.fields_f64_3[idx].data(),pdat.fields_f64_3[idx].size());
        }

        pdatbuf.fields_f64_3.push_back({std::move(buf),pdat.fields_f64_3[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_4.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_4>> buf;

        if(pdat.fields_f64_4[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_4>>(pdat.fields_f64_4[idx].data(),pdat.fields_f64_4[idx].size());
        }

        pdatbuf.fields_f64_4.push_back({std::move(buf),pdat.fields_f64_4[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_8.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_8>> buf;

        if(pdat.fields_f64_8[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_8>>(pdat.fields_f64_8[idx].data(),pdat.fields_f64_8[idx].size());
        }

        pdatbuf.fields_f64_8.push_back({std::move(buf),pdat.fields_f64_8[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_16.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_16>> buf;

        if(pdat.fields_f64_16[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_16>>(pdat.fields_f64_16[idx].data(),pdat.fields_f64_16[idx].size());
        }

        pdatbuf.fields_f64_16.push_back({std::move(buf),pdat.fields_f64_16[idx]});
    }





    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_u32.size(); idx++){
        std::unique_ptr<sycl::buffer<u32>> buf;

        if(pdat.fields_u32[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<u32>>(pdat.fields_u32[idx].data(),pdat.fields_u32[idx].size());
        }

        pdatbuf.fields_u32.push_back({std::move(buf),pdat.fields_u32[idx]});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_u64.size(); idx++){
        std::unique_ptr<sycl::buffer<u64>> buf;

        if(pdat.fields_u64[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<u64>>(pdat.fields_u64[idx].data(),pdat.fields_u64[idx].size());
        }

        pdatbuf.fields_u64.push_back({std::move(buf),pdat.fields_u64[idx]});
    }







    return pdatbuf;
}
