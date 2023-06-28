// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambindings/pybindaliases.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>

#include "NamedSetupSPH.hpp"

Register_pymod(pynamedsphsetup){

    py::class_<NamedSetupSPH>(m, "SetupSPH")
        .def(
            py::init([](std::string kernel, std::string precision) {
                return std::make_unique<NamedSetupSPH>(kernel, precision);
            }),
            py::kw_only(),
            py::arg("kernel"),
            py::arg("precision")
        )
        .def("init", &NamedSetupSPH::init)
        .def("set_boundaries", &NamedSetupSPH::set_boundaries)
        .def("get_box_dim", [](NamedSetupSPH & self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt){
            auto tmp = self.get_box_dim(dr, xcnt, ycnt, zcnt);
            return std::tuple{tmp.x(), tmp.y(), tmp.z()};
        })
        .def("get_ideal_box", [](NamedSetupSPH & self, f64 dr, std::tuple<f64,f64,f64> box_min, std::tuple<f64,f64,f64> box_max){
            
            auto [xm,ym,zm] = box_min;
            auto [xM,yM,zM] = box_max;

            auto [b1,b2] = self.get_ideal_box(dr, {f64_3{xm,ym,zm},f64_3{xM,yM,zM}});
            return std::tuple{std::tuple{b1.x(), b1.y(), b1.z()},std::tuple{b2.x(), b2.y(), b2.z()}};
        })
        .def("set_value_in_box", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::string type ,py::object val, std::string name, std::tuple<f64,f64,f64> box_min, std::tuple<f64,f64,f64> box_max){
            StackEntry stack_loc{};
            auto [xm,ym,zm] = box_min;
            auto [xM,yM,zM] = box_max;

            if(type == "f32"){
                f32 tmp = val.cast<f32>();
                self.set_value_in_box(ctx, tmp, name, {f64_3{xm,ym,zm},f64_3{xM,yM,zM}});
            }else if(type == "f64"){
                f64 tmp = val.cast<f64>();
                self.set_value_in_box(ctx, tmp, name, {f64_3{xm,ym,zm},f64_3{xM,yM,zM}});
            }else if(type == "f64_3"){
                auto tmp_ = val.cast<std::tuple<f64,f64,f64>>();
                f64_3 tmp {std::get<0>(tmp_), std::get<1>(tmp_), std::get<2>(tmp_)};
                self.set_value_in_box(ctx, tmp, name, {f64_3{xm,ym,zm},f64_3{xM,yM,zM}});
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown type");
            }


        })
        .def("set_value_in_sphere", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::string type ,py::object val, std::string name, std::tuple<f64,f64,f64> center, f64 radius){
            StackEntry stack_loc{};
            auto [xm,ym,zm] = center;

            if(type == "f32"){
                f32 tmp = val.cast<f32>();
                self.set_value_in_sphere(ctx, tmp, name, f64_3{xm,ym,zm},radius);
            }else if(type == "f64"){
                f64 tmp = val.cast<f64>();
                self.set_value_in_sphere(ctx, tmp, name, f64_3{xm,ym,zm},radius);
            }else if(type == "f64_3"){
                auto tmp_ = val.cast<std::tuple<f64,f64,f64>>();
                f64_3 tmp {std::get<0>(tmp_), std::get<1>(tmp_), std::get<2>(tmp_)};
                self.set_value_in_sphere(ctx, tmp, name, f64_3{xm,ym,zm},radius);
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown type");
            }


        })
        .def("get_sum_float", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::string type, std::string name){
            StackEntry stack_loc{};

            f64 ret_val;

            if(type == "f32"){
                ret_val = self.get_sum<f32>(ctx, name);
            }else if(type == "f64"){
                ret_val = self.get_sum<f64>(ctx, name);
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown type");
            }

            return ret_val;
        })
        .def("get_sum_vec", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::string type, std::string name){
            StackEntry stack_loc{};

            std::tuple<f64,f64,f64> ret_val;

            if(type == "f64_3"){
                f64_3 tmp = self.get_sum<f64_3>(ctx, name);
                std::get<0>(ret_val) = tmp.x();
                std::get<1>(ret_val) = tmp.y();
                std::get<2>(ret_val) = tmp.z();
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>("unknown type");
            }

            return ret_val;
        })
        .def("pertub_eigenmode_wave", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::tuple<f64,f64> ampls, std::tuple<f64,f64,f64> k, f64 phase){
            auto [kx,ky,kz] = k;
            self.pertub_eigenmode_wave(ctx, ampls, {kx,ky,kz}, phase);
        })
        .def("add_particules_fcc", [](NamedSetupSPH & self, ShamrockCtx & ctx, f64 dr, std::tuple<f64,f64,f64> box_min, std::tuple<f64,f64,f64> box_max){
            auto [xm,ym,zm] = box_min;
            auto [xM,yM,zM] = box_max;
            self.add_particules_fcc(ctx, dr, {f64_3{xm,ym,zm},f64_3{xM,yM,zM}});
        })
        .def("set_total_mass", &NamedSetupSPH::set_total_mass)
        .def("get_part_mass", &NamedSetupSPH::get_part_mass)
        .def("update_smoothing_lenght", &NamedSetupSPH::update_smoothing_lenght)
        .def("get_closest_part_to", [](NamedSetupSPH & self, ShamrockCtx & ctx, std::tuple<f64,f64,f64> pos) -> std::tuple<f64,f64,f64> {
            auto [x,y,z] = pos;
            auto tmp = self.get_closest_part_to(ctx, {x,y,z});
            return {tmp.x(),tmp.y(),tmp.z()};
        })
        
        
        
        ;


}