// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shammodels/sph/legacy/models/basic_sph_gas_uint.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <variant>


class NamedBasicSPHUinterne{
    
    
    using var_t = std::variant<
        models::sph::BasicSPHGasUInterne<f32, shamrock::sph::kernels::M4<f32>>,
        models::sph::BasicSPHGasUInterne<f32, shamrock::sph::kernels::M6<f32>>
    >;

    var_t model;

    public:

    NamedBasicSPHUinterne(std::string kernel_name, std::string precision){
        if(kernel_name == "M4" && precision == "single"){
            model = models::sph::BasicSPHGasUInterne<f32, shamrock::sph::kernels::M4<f32>>{};
        }else if(kernel_name == "M6" && precision == "single"){
            model = models::sph::BasicSPHGasUInterne<f32, shamrock::sph::kernels::M6<f32>>{};
        }else{
            std::invalid_argument("unknown configuration");
        }
    }

    inline void init(){
        std::visit([&](auto && arg) {
            arg.init();
        }, model);
    }

    inline f64 evolve(ShamrockCtx &ctx, f64 current_time, f64 target_time){
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        return std::visit([&](auto && arg) {
            return arg.evolve(*ctx.sched,current_time,target_time);
        }, model);
    }

    inline f64 simulate_until(ShamrockCtx &ctx, f64 start_time, f64 end_time, u32 freq_dump, u32 freq_restart_dump,std::string prefix_dump){
        if(!ctx.sched){
            throw std::runtime_error("cannot initialize a setup with an uninitialized scheduler");
        }

        return std::visit([&](auto && arg) {
            return arg.simulate_until(*ctx.sched,start_time, end_time, freq_dump, freq_restart_dump,prefix_dump);
        }, model);
    }
    inline void close(){
        std::visit([&](auto && arg) {
            arg.close();
        }, model);
    }


    inline void set_cfl_cour(f64 Ccour){
        std::visit([&](auto && arg) {
            arg.set_cfl_cour(Ccour);
        }, model);
    }
    inline void set_cfl_force(f64 Cforce){
        std::visit([&](auto && arg) {
            arg.set_cfl_force(Cforce);
        }, model);
    }
    inline void set_particle_mass(f64 pmass){
        std::visit([&](auto && arg) {
            arg.set_particle_mass(pmass);
        }, model);
    }

};