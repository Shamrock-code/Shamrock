// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "interface_generator.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
//#include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_patch_data.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"

#include "interface_generator_impl.hpp"


//TODO can merge those 2 func

template <>
std::vector<std::unique_ptr<shamrock::patch::PatchData>> InterfaceVolumeGenerator::append_interface<f32_3>(sycl::queue &queue, shamrock::patch::PatchData & pdat,
                                                                        std::vector<f32_3> boxs_min,
                                                                        std::vector<f32_3> boxs_max,f32_3 add_offset) {


    using namespace shamrock::patch;

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.pdl);
    }

    std::vector<std::vector<u32>> idxs(boxs_min.size());

    for (u32 i = 0; i < flag_choice.size(); i++) {
        if(flag_choice[i] < boxs_min.size()){
            idxs[flag_choice[i]].push_back(i);
        }
    }


    if (! pdat.is_empty()) {
        for (u32 i = 0; i < idxs.size(); i++) {
            pdat.append_subset_to(idxs[i], *pdat_vec[i]);
            u32 ixyz = pdat.pdl.get_field_idx<f32_3>("xyz");
            pdat_vec[i]->get_field<f32_3>(ixyz).apply_offset(add_offset);
        }
    }

    

    return pdat_vec;

}

template <>
std::vector<std::unique_ptr<shamrock::patch::PatchData>> InterfaceVolumeGenerator::append_interface<f64_3>(sycl::queue &queue, shamrock::patch::PatchData & pdat,
                                                                        std::vector<f64_3> boxs_min,
                                                                        std::vector<f64_3> boxs_max,f64_3 add_offset) {
using namespace shamrock::patch;

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.pdl);
    }

    std::vector<std::vector<u32>> idxs(boxs_min.size());

    for (u32 i = 0; i < flag_choice.size(); i++) {
        if(flag_choice[i] < boxs_min.size()){
            idxs[flag_choice[i]].push_back(i);
        }
    }

    if (! pdat.is_empty()) {
        for (u32 i = 0; i < idxs.size(); i++) {
            pdat.append_subset_to(idxs[i], *pdat_vec[i]);
            u32 ixyz = pdat.pdl.get_field_idx<f64_3>("xyz");
            pdat_vec[i]->get_field<f64_3>(ixyz).apply_offset(add_offset);
        }
    }

    

    return pdat_vec;

}




