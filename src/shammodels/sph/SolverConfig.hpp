// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include <shamunits/UnitSystem.hpp>
#include <shamunits/Constants.hpp>
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <variant>

#include "shambackends/typeAliasVec.hpp"

namespace shammodels::sph {
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;
}

template<class Tvec, template<class> class SPHKernel>
struct shammodels::sph::SolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    static constexpr Tscal Rkern = Kernel::Rkern;

    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    struct AVConfig {

        /**
         * @brief cf Price 2018 , q^a_ab = 0
         */
        struct None {};

        struct Constant {
            Tscal alpha_u  = 1.0;
            Tscal alpha_AV = 1.0;
            Tscal beta_AV  = 2.0;
        };

        /**
         * @brief Morris & Monaghan 1997
         *
         */
        struct VaryingMM97 {
            Tscal alpha_min   = 0.1;
            Tscal alpha_max   = 1.0;
            Tscal sigma_decay = 0.1;
            Tscal alpha_u     = 1.0;
            Tscal beta_AV     = 2.0;
        };

        /**
         * @brief Cullen & Dehnen 2010
         *
         */
        struct VaryingCD10 {
            Tscal alpha_min   = 0.1;
            Tscal alpha_max   = 1.0;
            Tscal sigma_decay = 0.1;
            Tscal alpha_u     = 1.0;
            Tscal beta_AV     = 2.0;
        };

        using Variant  = std::variant<None, Constant, VaryingMM97, VaryingCD10>;
        Variant config = Constant{};

        void set(Variant v) { config = v; }

        inline bool has_alphaAV_field() {
            bool is_varying_alpha =
                bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
            return is_varying_alpha;
        }

        inline bool has_divv_field() {
            bool is_varying_alpha =
                bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
            return is_varying_alpha;
        }
        inline bool has_curlv_field() {
            bool is_varying_alpha =
                bool(std::get_if<VaryingCD10>(&config));
            return is_varying_alpha;
        }
        inline bool has_dtdivv_field() {
            bool is_varying_alpha =
                bool(std::get_if<VaryingCD10>(&config));
            return is_varying_alpha;
        }

        inline bool has_field_soundspeed() {

            // this should not be needed idealy, but we need the pressure on the ghosts and 
            // we don't want to communicate it as it can be recomputed from the other fields
            // hence we copy the soundspeed at the end of the step to a field in the patchdata
            // cf eos module there is another soundspeed field available as a Compute field
            // unifying the patchdata and the ghosts is really needed ...
            
            bool is_varying_alpha =
                bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
            return is_varying_alpha;
        }

        inline void print_status() {
            logger::raw_ln("--- artificial viscosity config");

            if (None *v = std::get_if<None>(&config)) {
                logger::raw_ln("  Config Type : None (No artificial viscosity)");
            } else if (Constant *v = std::get_if<Constant>(&config)) {
                logger::raw_ln("  Config Type : Constant (Constant artificial viscosity)");
                logger::raw_ln("  alpha_u  =", v->alpha_u);
                logger::raw_ln("  alpha_AV =", v->alpha_AV);
                logger::raw_ln("  beta_AV  =", v->beta_AV);
            } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&config)) {
                logger::raw_ln("  Config Type : VaryingMM97 (Morris & Monaghan 1997)");
                logger::raw_ln("  alpha_min   =", v->alpha_min);
                logger::raw_ln("  alpha_max   =", v->alpha_max);
                logger::raw_ln("  sigma_decay =", v->sigma_decay);
                logger::raw_ln("  alpha_u     =", v->alpha_u);
                logger::raw_ln("  beta_AV     =", v->beta_AV);
            } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&config)) {
                logger::raw_ln("  Config Type : VaryingCD10 (Cullen & Dehnen 2010)");
                logger::raw_ln("  alpha_min   =", v->alpha_min);
                logger::raw_ln("  alpha_max   =", v->alpha_max);
                logger::raw_ln("  sigma_decay =", v->sigma_decay);
                logger::raw_ln("  alpha_u     =", v->alpha_u);
                logger::raw_ln("  beta_AV     =", v->beta_AV);
            }

            logger::raw_ln("--- artificial viscosity config (deduced)");

            logger::raw_ln("-------------");
        }
    };


    struct BCConfig{
        struct Free{
            Tscal expand_tolerance = 1.2;
        };
        struct Periodic{

        };
        struct ShearingPeriodic{
            i32_3 shear_base; 
            i32_3 shear_dir; 
            Tscal shear_speed;
        };

        using Variant = std::variant<Free,Periodic,ShearingPeriodic>;

        Variant config = Free{};

        inline void set_free(){
            config = Free{};
        }

        inline void set_periodic(){
            config = Periodic{};
        }

        inline void set_shearing_periodic(
            i32_3 shear_base,
            i32_3 shear_dir, Tscal speed){
            config = ShearingPeriodic{
                shear_base,shear_dir,speed
            };
        }
    };  

    BCConfig boundary_config;

    inline void set_boundary_free(){
       boundary_config.set_free();
    }

    inline void set_boundary_periodic(){
        boundary_config.set_periodic();
    }

    inline void set_boundary_shearing_periodic(
            i32_3 shear_base,
            i32_3 shear_dir, Tscal speed){
        boundary_config.set_shearing_periodic(shear_base,shear_dir, speed);
    }


    AVConfig artif_viscosity;

    inline void set_artif_viscosity_None() {
        using Tmp = typename AVConfig::None;
        artif_viscosity.set(Tmp{});
    }

    inline void set_artif_viscosity_Constant(typename AVConfig::Constant v) {
        artif_viscosity.set(v);
    }

    inline void set_artif_viscosity_VaryingMM97(typename AVConfig::VaryingMM97 v) {
        artif_viscosity.set(v);
    }

    inline void set_artif_viscosity_VaryingCD10(typename AVConfig::VaryingCD10 v) {
        artif_viscosity.set(v);
    }

    inline bool has_field_uint() {
        // no barotropic for now
        return true;
    }

    inline bool has_field_alphaAV() { return artif_viscosity.has_alphaAV_field(); }

    inline bool has_field_divv() { return artif_viscosity.has_alphaAV_field(); }
    inline bool has_field_dtdivv() { return artif_viscosity.has_dtdivv_field(); }
    inline bool has_field_curlv() { return artif_viscosity.has_curlv_field() && (dim == 3); }

    inline bool has_field_soundspeed() { return artif_viscosity.has_field_soundspeed(); }

    inline void print_status() { 
        if(shamcomm::world_rank() != 0){return;}
        logger::raw_ln("----- SPH Solver configuration -----");
        
        artif_viscosity.print_status(); 
        
        
        logger::raw_ln("------------------------------------");
    }


    inline void set_units(shamunits::UnitSystem<Tscal> new_sys){
        unit_sys = new_sys;
    }

    inline Tscal get_constant_G(){
        if(!unit_sys){
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        }else{
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }
};
