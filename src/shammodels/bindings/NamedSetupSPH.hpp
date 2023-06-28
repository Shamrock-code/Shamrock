// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shammodels/sph/legacy/setup/sph_setup.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <variant>



template<class T>
sycl::vec<f64, 3> convert_vec(T a);

template<>
sycl::vec<f64, 3> convert_vec(sycl::vec<f64, 3> a){
    return sycl::vec<f64, 3> {a.x(), a.y(), a.z()};
}

template<>
sycl::vec<f64, 3> convert_vec(sycl::vec<f32, 3> a){
    return sycl::vec<f64, 3> {a.x(), a.y(), a.z()};
}




class NamedSetupSPH{
    
    
    using var_t = std::variant<
        models::sph::SetupSPH<f32, shamrock::sph::kernels::M4<f32>>,
        models::sph::SetupSPH<f32, shamrock::sph::kernels::M6<f32>>,
        models::sph::SetupSPH<f64, shamrock::sph::kernels::M4<f64>>
    >;

    var_t setup;

    public:

    using vec = sycl::vec<f64,3>;

    NamedSetupSPH(std::string kernel_name, std::string precision){
        if(kernel_name == "M4" && precision == "single"){
            setup = models::sph::SetupSPH<f32, shamrock::sph::kernels::M4<f32>>{};
        }else if(kernel_name == "M6" && precision == "single"){
            setup = models::sph::SetupSPH<f32, shamrock::sph::kernels::M6<f32>>{};
        }else if(kernel_name == "M4" && precision == "double"){
            setup = models::sph::SetupSPH<f64, shamrock::sph::kernels::M4<f64>>{};
        }else{
            throw shambase::throw_with_loc<std::invalid_argument>("unknown configuration");
        }
    }


    void init(ShamrockCtx & ctx){

        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        std::visit([&](auto && arg) {
            arg.init(*ctx.sched);
        }, setup);
    }

    void set_boundaries(std::string type){
        if(type == "periodic"){
            std::visit([=](auto && arg) {
                arg.set_boundaries(true);
            }, setup);
        }else if(type == "free"){
            std::visit([=](auto && arg) {
                arg.set_boundaries(false);
            }, setup);
        }else{
            throw std::invalid_argument("this type of boundary is unknown");
        }
    }


    inline vec get_box_dim(f64 dr, u32 xcnt, u32 ycnt, u32 zcnt){
        return std::visit([&](auto && arg) {
            return convert_vec(arg.get_box_dim(dr, xcnt, ycnt, zcnt));
        }, setup);
    }

    inline std::tuple<vec,vec> get_ideal_box(f64 dr, std::tuple<vec,vec> box){
        vec b1 = std::get<0>(box);
        vec b2 = std::get<1>(box);
        return std::visit([&](auto && arg) {
            auto [a,b] = arg.get_ideal_box(dr, {{b1.x(), b1.y(), b1.z()},{b2.x(), b2.y(), b2.z()}});
            return std::tuple<vec,vec>{convert_vec(a),convert_vec(b)};
        }, setup);
    }

    template<class T> 
    inline void set_value_in_box(ShamrockCtx & ctx, T val, std::string name, std::tuple<vec,vec> box){
        
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }
        
        vec b1 = std::get<0>(box);
        vec b2 = std::get<1>(box);

        std::visit([&](auto && arg) {
            arg.set_value_in_box(*ctx.sched, val, name, {{b1.x(), b1.y(), b1.z()},{b2.x(), b2.y(), b2.z()}});
        }, setup);
    }

    template<class T> 
    inline void set_value_in_sphere(ShamrockCtx & ctx, T val, std::string name, vec center, f64 radius){
        
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        std::visit([&](auto && arg) {
            arg.set_value_in_sphere(*ctx.sched, val, name, {center.x(), center.y(), center.z()},radius);
        }, setup);
    }

    template<class T> 
    inline T get_sum(ShamrockCtx & ctx, std::string name){
        
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        return std::visit([&](auto && arg) {
            return arg.template get_sum<T>(*ctx.sched, name);
        }, setup);
    }

    inline void pertub_eigenmode_wave(ShamrockCtx & ctx, std::tuple<f64,f64> ampls, vec k, f64 phase){

        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        std::visit([&](auto && arg) {
            arg.pertub_eigenmode_wave(*ctx.sched, {std::get<0>(ampls),std::get<1>(ampls)}, {k.x(), k.y(), k.z()}, phase);
        }, setup);
    }


    inline void add_particules_fcc(ShamrockCtx & ctx, f64 dr, std::tuple<vec,vec> box){

        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        vec b1 = std::get<0>(box);
        vec b2 = std::get<1>(box);

        std::visit([&](auto && arg) {
            arg.add_particules_fcc(*ctx.sched, dr, {{b1.x(), b1.y(), b1.z()},{b2.x(), b2.y(), b2.z()}});
        }, setup);

    }        
    
    inline vec get_closest_part_to(ShamrockCtx & ctx,vec pos){
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        vec ret;

        std::visit([&](auto && arg) {
            auto tmp = arg.get_closest_part_to(*ctx.sched, {pos.x(),pos.y(),pos.z()});
            ret.x() = tmp.x();
            ret.y() = tmp.y();
            ret.z() = tmp.z();
        }, setup);

        return ret;
    }

    inline void set_total_mass(f64 tot_mass){
        std::visit([&](auto && arg) {
            arg.set_total_mass(tot_mass);
        }, setup);
    }

    inline f64 get_part_mass(){
        return std::visit([&](auto && arg) {
            return f64(arg.get_part_mass());
        }, setup);
    }

    void update_smoothing_lenght(ShamrockCtx & ctx){
        StackEntry stack_loc{};
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        std::visit([&](auto && arg) {
            arg.update_smoothing_lenght(*ctx.sched);
        }, setup);
    }

};