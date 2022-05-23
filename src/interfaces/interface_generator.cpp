#include "interface_generator.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include "utils/geometry_utils.hpp"

#include "interface_generator_impl.hpp"


//TODO can merge those 2 func

template <>
std::vector<std::unique_ptr<PatchData>> InterfaceVolumeGenerator::append_interface<f32_3>(sycl::queue &queue, PatchData & pdat,
                                                                        std::vector<f32_3> boxs_min,
                                                                        std::vector<f32_3> boxs_max,f32_3 add_offset) {


    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.patchdata_layout);
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
        }
    }

    return pdat_vec;

}

template <>
std::vector<std::unique_ptr<PatchData>> InterfaceVolumeGenerator::append_interface<f64_3>(sycl::queue &queue, PatchData & pdat,
                                                                        std::vector<f64_3> boxs_min,
                                                                        std::vector<f64_3> boxs_max,f64_3 add_offset) {

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>(pdat.patchdata_layout);
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
        }
    }

    return pdat_vec;

}




