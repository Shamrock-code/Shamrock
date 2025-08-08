// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file merged_patch.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/patch/interfaces/interface_handler.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

template<class flt>
class MergedPatchData {
    public:
    using vec = sycl::vec<flt, 3>;

    u32 or_element_cnt = 0;
    shamrock::patch::PatchDataLayer data;
    std::tuple<vec, vec> box;

    MergedPatchData(const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> & pdl) : data(pdl) {};

    [[nodiscard]]
    static std::unordered_map<u64, MergedPatchData<flt>>
    merge_patches(PatchScheduler &sched, LegacyInterfacehandler<vec, flt> &interface_hndl);

    inline void write_back(shamrock::patch::PatchDataLayer &pdat) {
        pdat.overwrite(data, or_element_cnt);
    }
};

template<class flt>
inline void write_back_merge_patches(
    PatchScheduler &sched, std::unordered_map<u64, MergedPatchData<flt>> &merge_pdat) {

    using namespace shamrock::patch;
    shamlog_debug_sycl_ln("Merged Patch", "write back merged buffers");

    sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
        if (merge_pdat.at(id_patch).or_element_cnt == 0)
            std::cout << " empty => skipping" << std::endl;

        shamlog_debug_sycl_ln("Merged Patch", "patch : n°", id_patch, "-> write back merge buf");

        merge_pdat.at(id_patch).write_back(pdat);
    });
}

template<class flt, class T>
class MergedPatchCompField {
    public:
    using vec = sycl::vec<flt, 3>;

    u32 or_element_cnt = 0;
    PatchDataField<T> buf;

    MergedPatchCompField() : buf("comp_field", 1) {};

    [[nodiscard]]
    static std::unordered_map<u64, MergedPatchCompField<flt, T>> merge_patches_cfield(
        PatchScheduler &sched,
        LegacyInterfacehandler<vec, flt> &interface_hndl,
        PatchComputeField<T> &comp_field,
        PatchComputeFieldInterfaces<T> &comp_field_interf);

    inline void write_back(PatchDataField<T> &field) { field.overwrite(buf, or_element_cnt); }
};

#if false

template<class vec>

struct [[deprecated]] MergedPatchDataBuffer {public:
    u32 or_element_cnt;
    std::unique_ptr<PatchDataBuffer> data;
    std::tuple<vec,vec> box;
};

template<class T>
struct [[deprecated]] MergedPatchCompFieldBuffer {public:
    u32 or_element_cnt;
    std::unique_ptr<sycl::buffer<T>> buf;
};



template<class pos_prec,class pos_vec>
[[deprecated]]
inline void make_merge_patches(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,

    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){

    shamlog_debug_sycl_ln("Merged Patch","make_merge_patches");

    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {



        auto tmp_box = sched.patch_data.sim_box.get_box<pos_prec>(cur_p);

        f32_3 min_box = std::get<0>(tmp_box);
        f32_3 max_box = std::get<1>(tmp_box);

        shamlog_debug_sycl_ln("Merged Patch","patch : n°",id_patch , "-> making merge buf");

        u32 len_main = pdat_buf.element_count;

        u32 original_element = len_main;
        //merge_pdat_buf[id_patch].or_element_cnt = len_main;

        {
            const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst = interface_hndl.get_interface_list(id_patch);
            for (auto & [int_pid, pdat_ptr] : p_interf_lst) {


                u32 cnt = pdat_ptr->get_obj_cnt();
                //std::cout << "received interf : " << cnt << std::endl;
                len_main += (cnt);
            }
        }

        u32 total_element = len_main;
        //merge_pdat_buf[id_patch].data.element_count = len_main;




        std::unique_ptr<PatchDataBuffer> merged_buf = std::make_unique<PatchDataBuffer>(pdat_buf.pdl, total_element);



        std::vector<u32> fields_f32_offset;
        std::vector<u32> fields_f32_2_offset;
        std::vector<u32> fields_f32_3_offset;
        std::vector<u32> fields_f32_4_offset;
        std::vector<u32> fields_f32_8_offset;
        std::vector<u32> fields_f32_16_offset;
        std::vector<u32> fields_f64_offset;
        std::vector<u32> fields_f64_2_offset;
        std::vector<u32> fields_f64_3_offset;
        std::vector<u32> fields_f64_4_offset;
        std::vector<u32> fields_f64_8_offset;
        std::vector<u32> fields_f64_16_offset;
        std::vector<u32> fields_u32_offset;
        std::vector<u32> fields_u64_offset;







        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32.size(); idx++){
            fields_f32_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_2.size(); idx++){
            fields_f32_2_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_3.size(); idx++){
            fields_f32_3_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_4.size(); idx++){
            fields_f32_4_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_8.size(); idx++){
            fields_f32_8_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_16.size(); idx++){
            fields_f32_16_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64.size(); idx++){
            fields_f64_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_2.size(); idx++){
            fields_f64_2_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_3.size(); idx++){
            fields_f64_3_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_4.size(); idx++){
            fields_f64_4_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_8.size(); idx++){
            fields_f64_8_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_16.size(); idx++){
            fields_f64_16_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_u32.size(); idx++){
            fields_u32_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_u64.size(); idx++){
            fields_u64_offset.push_back(0);
        }

    #define X(arg)                                                                                 \
        for (u32 idx = 0; idx < pdat_buf.pdl.fields_##arg.size(); idx++) {                         \
            u32 nvar = merged_buf->pdl.fields_##arg[idx].nvar;                                     \
            syclalgs::basic::write_with_offset_into(                                               \
                *merged_buf->fields_##arg[idx],                                                    \
                *pdat_buf.fields_##arg[idx],                                                       \
                0,                                                                                 \
                pdat_buf.element_count *nvar);                                                     \
            fields_##arg##_offset[idx] += pdat_buf.element_count * nvar;                           \
        }
        XMAC_LIST_ENABLED_FIELD
    #undef X










        interface_hndl.for_each_interface_buf(
            id_patch,
            shamsys::instance::get_compute_queue(),
            [&](u64 patch_id, u64 interf_patch_id, PatchDataBuffer & interfpdat, std::tuple<f32_3,f32_3> box){

                //std::cout <<  "patch : n°"<< id_patch << " -> interface : "<<interf_patch_id << " merging" << std::endl;

                min_box = sycl::min(std::get<0>(box),min_box);
                max_box = sycl::max(std::get<1>(box),max_box);

    #define X(arg)                                                                                 \
        for (u32 idx = 0; idx < interfpdat.pdl.fields_##arg.size(); idx++) {                       \
            u32 nvar = merged_buf->pdl.fields_##arg[idx].nvar;                                     \
            syclalgs::basic::write_with_offset_into(                                               \
                *merged_buf->fields_##arg[idx],                                                    \
                *interfpdat.fields_##arg[idx],                                                     \
                fields_##arg##_offset[idx],                                                        \
                interfpdat.element_count *nvar);                                                   \
            fields_##arg##_offset[idx] += interfpdat.element_count * nvar;                         \
        }
                XMAC_LIST_ENABLED_FIELD
    #undef X


            }
        );

        merge_pdat_buf[id_patch].or_element_cnt = original_element;
        merge_pdat_buf[id_patch].data = std::move(merged_buf);
        merge_pdat_buf[id_patch].box = {min_box,max_box};




    });


}



template<class pos_prec,class pos_vec>
[[deprecated]]
inline void write_back_merge_patches(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,

    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){


    shamlog_debug_sycl_ln("Merged Patch","write back merged buffers");



    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
        if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;


        shamlog_debug_sycl_ln("Merged Patch","patch : n°",id_patch , "-> write back merge buf");

    #define X(arg)                                                                                 \
        for (u32 idx = 0; idx < pdat_buf.pdl.fields_##arg.size(); idx++) {                         \
            u32 nvar = pdat_buf.pdl.fields_##arg[idx].nvar;                                        \
            syclalgs::basic::write_with_offset_into(                                               \
                *pdat_buf.fields_##arg[idx],                                                       \
                *merge_pdat_buf.at(id_patch).data->fields_##arg[idx],                              \
                0,                                                                                 \
                pdat_buf.element_count *nvar);                                                     \
        }
        XMAC_LIST_ENABLED_FIELD
    #undef X


    });

}



template<class pos_prec,class pos_vec,class T>
[[deprecated]]
inline void make_merge_patches_comp_field(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,

    PatchComputeField<f32> & comp_field,
    PatchComputeFieldInterfaces<f32> & comp_field_interf,

    std::unordered_map<u64,MergedPatchCompFieldBuffer<T>> & merge_pdat_comp_field){

    shamlog_debug_sycl_ln("Merged Patch","make_merge_patches_comp_field");

    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {



        auto compfield_buf = comp_field.get_sub_buf(id_patch);

        shamlog_debug_sycl_ln("Merged Patch","patch : n°",id_patch , "-> making merge comp field");

        u32 len_main = compfield_buf->size();// TODO remove ref to size
        merge_pdat_comp_field[id_patch].or_element_cnt = len_main;

        {

            const std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>> & p_interf_lst = comp_field_interf.interface_map[id_patch];
            for (auto & [int_pid, pdat_ptr] : p_interf_lst) {
                len_main += (pdat_ptr->size());
            }
        }


        merge_pdat_comp_field[id_patch].buf = std::make_unique<sycl::buffer<f32>>(len_main);


        u32 offset_buf = 0;


        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            auto source = compfield_buf->get_access<sycl::access::mode::read>(cgh);
            auto dest = merge_pdat_comp_field[id_patch].buf->template get_access<sycl::access::mode::discard_write>(cgh);
            cgh.parallel_for( sycl::range{
                compfield_buf->size() // TODO remove ref to size
            }, [=](sycl::item<1> item) { dest[item] = source[item]; });
        });
        offset_buf += compfield_buf->size();// TODO remove ref to size


        std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>> & p_interf_lst = comp_field_interf.interface_map[id_patch];

        for (auto & [int_pid, pdat_ptr] : p_interf_lst) {

            if(pdat_ptr->size() > 0){

                //std::cout <<  "patch : n°"<< id_patch << " -> interface : "<<interf_patch_id << " merging" << std::endl;
                auto tmp_buf = pdat_ptr->get_sub_buf();

                u32 len_int =  pdat_ptr->size();

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    auto source = tmp_buf->template get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_comp_field[id_patch].buf->template get_access<sycl::access::mode::discard_write>(cgh);
                    auto off = offset_buf;
                    cgh.parallel_for( sycl::range{len_int}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                });
                offset_buf += len_int;

            }
        }






    });


}
#endif
