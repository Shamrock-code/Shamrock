// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SimBox.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamrock/patch/SimBox.hpp"

// from cppreference std::visit page
// helper type for the visitor #4
template<class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


std::array<f32, 3> to_array(f32_3 v){
    return {v[0],v[1],v[2]};
}
std::array<f64, 3> to_array(f64_3 v){
    return {v[0],v[1],v[2]};
}
std::array<u32, 3> to_array(u32_3 v){
    return {v[0],v[1],v[2]};
}
std::array<u64, 3> to_array(u64_3 v){
    return {v[0],v[1],v[2]};
}
std::array<i64, 3> to_array(i64_3 v){
    return {v[0],v[1],v[2]};
}

f32_3 to_sycl_vec(std::array<f32,3> v){
    return {v[0],v[1],v[2]};
}
f64_3 to_sycl_vec(std::array<f64,3> v){
    return {v[0],v[1],v[2]};
}
u32_3 to_sycl_vec(std::array<u32,3> v){
    return {v[0],v[1],v[2]};
}
u64_3 to_sycl_vec(std::array<u64,3> v){
    return {v[0],v[1],v[2]};
}
i64_3 to_sycl_vec(std::array<i64,3> v){
    return {v[0],v[1],v[2]};
}


namespace shamrock::patch {

    void SimulationBoxInfo::to_json(nlohmann::json &j) {
        auto &pcoord           = patch_coord_bounding_box;
        auto &bounding_box_var = bounding_box;

        j["patchcoormin"] = pcoord.coord_min;
        j["patchcoormax"] = pcoord.coord_max;

        std::visit(
            overloaded{
                [&](auto arg) {
                    shambase::throw_unimplemented();
                },
                [&](shammath::CoordRange<f32_3> arg) {
                    j["coordtype"] = "f32_3";
                    j["box_min"] = to_array(arg.lower);
                    j["box_max"] = to_array(arg.upper);
                },
                [&](shammath::CoordRange<f64_3> arg) {
                    j["coordtype"] = "f64_3";
                    j["box_min"] = to_array(arg.lower);
                    j["box_max"] = to_array(arg.upper);
                },
                [&](shammath::CoordRange<u32_3> arg) {
                    j["coordtype"] = "u32_3";
                    j["box_min"] = to_array(arg.lower);
                    j["box_max"] = to_array(arg.upper);
                },
                [&](shammath::CoordRange<u64_3> arg) {
                    j["coordtype"] = "u64_3";
                    j["box_min"] = to_array(arg.lower);
                    j["box_max"] = to_array(arg.upper);
                },
                [&](shammath::CoordRange<i64_3> arg) {
                    j["coordtype"] = "i64_3";
                    j["box_min"] = to_array(arg.lower);
                    j["box_max"] = to_array(arg.upper);
                }},
            bounding_box_var.value);
    }

    void SimulationBoxInfo::from_json(const nlohmann::json &j) {
        PatchCoord<3> pcoord;
        j.at("patchcoormin").get_to(pcoord.coord_min);
        j.at("patchcoormax").get_to(pcoord.coord_max);
        patch_coord_bounding_box = pcoord;

        logger::raw_ln(pdl.check_main_field_type<f64_3>(),j.at("coordtype").get<std::string>(),j.at("coordtype").get<std::string>() == "f64_3");
        if (pdl.check_main_field_type<f32_3>() && (j.at("coordtype").get<std::string>() == "f32_3")) {
            bounding_box.value = shammath::CoordRange<f32_3>{
                to_sycl_vec(j.at("box_min").get<std::array<f32,3>>()),
                to_sycl_vec(j.at("box_max").get<std::array<f32,3>>()),
            };
        } else if (pdl.check_main_field_type<f64_3>() && (j.at("coordtype").get<std::string>() == "f64_3")) {
            bounding_box.value = shammath::CoordRange<f64_3>{
                to_sycl_vec(j.at("box_min").get<std::array<f64,3>>()),
                to_sycl_vec(j.at("box_max").get<std::array<f64,3>>()),
            };
        } else if (pdl.check_main_field_type<u32_3>() && (j.at("coordtype").get<std::string>() == "u32_3")) {
            bounding_box.value = shammath::CoordRange<u32_3>{
                to_sycl_vec(j.at("box_min").get<std::array<u32,3>>()),
                to_sycl_vec(j.at("box_max").get<std::array<u32,3>>()),
            };
        } else if (pdl.check_main_field_type<u64_3>() && (j.at("coordtype").get<std::string>() == "u64_3")) {
            bounding_box.value = shammath::CoordRange<u64_3>{
                to_sycl_vec(j.at("box_min").get<std::array<u64,3>>()),
                to_sycl_vec(j.at("box_max").get<std::array<u64,3>>()),
            };
        } else if (pdl.check_main_field_type<i64_3>() && (j.at("coordtype").get<std::string>() == "i64_3")) {
            bounding_box.value = shammath::CoordRange<i64_3>{
                to_sycl_vec(j.at("box_min").get<std::array<i64,3>>()),
                to_sycl_vec(j.at("box_max").get<std::array<i64,3>>()),
            };
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>("unable to parse json type");
        }
    }

} // namespace shamrock::patch
