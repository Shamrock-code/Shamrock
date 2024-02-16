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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/ExtForceConfig.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <variant>

#include "config/AVConfig.hpp"
#include "config/BCConfig.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammodels/EOSConfig.hpp"

namespace shammodels::sph {
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    template<class Tvec>
    struct SolverStatusVar;

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::SolverStatusVar {

    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time   = 0;
    Tscal dt_sph = 0;

    Tscal cfl_multiplier = 1e-2;
};

template<class Tvec, template<class> class SPHKernel>
struct shammodels::sph::SolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    static constexpr Tscal Rkern = Kernel::Rkern;


    Tscal gpart_mass;
    Tscal cfl_cour;
    Tscal cfl_force;
    Tscal cfl_multiplier_stiffness = 2;


    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }

    inline Tscal get_constant_G() {
        if (!unit_sys) {
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }

    inline Tscal get_constant_c() {
        if (!unit_sys) {
            logger::warn_ln("sph::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.c();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.c();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables
    //////////////////////////////////////////////////////////////////////////////////////////////

    using SolverStatusVar = SolverStatusVar<Tvec>;

    SolverStatusVar time_state;

    inline void set_time(Tscal t) { time_state.time = t; }
    inline void set_next_dt(Tscal dt) { time_state.dt_sph = dt; }

    inline Tscal get_time() { return time_state.time; }
    inline Tscal get_dt_sph() { return time_state.dt_sph; }

    inline void set_cfl_multipler(Tscal lambda){time_state.cfl_multiplier = lambda;}
    inline Tscal get_cfl_multipler(){return time_state.cfl_multiplier;}

    inline void set_cfl_mult_stiffness(Tscal cstiff){cfl_multiplier_stiffness = cstiff;}
    inline Tscal get_cfl_mult_stiffness(){return cfl_multiplier_stiffness;}

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;
    bool use_two_stage_search = true;
    u64 max_neigh_cache_size = 10e9;

    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }
    inline void set_max_neigh_cache_size(u64 val) { max_neigh_cache_size = val; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool combined_dtdiv_divcurlv_compute = false;
    Tscal htol_up_tol   = 1.1;
    Tscal htol_up_iter  = 1.1;
    Tscal epsilon_h = 1e-6;
    u32 h_iter_per_subcycles = 50;
    u32 h_max_subcycles_count = 100; 

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using EOSConfig = shammodels::EOSConfig<Tvec>;

    EOSConfig eos_config;

    inline bool is_eos_locally_isothermal() {
        using T = typename EOSConfig::LocallyIsothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }
    inline void set_eos_locally_isothermal() { eos_config.set_locally_isothermal(); }
    inline void set_eos_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) { eos_config.set_locally_isothermalLP07(cs0, q, r0); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using AVConfig = AVConfig<Tvec>;

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
    inline void set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) {
        artif_viscosity.set(v);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using BCConfig       = BCConfig<Tvec>;

    BCConfig boundary_config;

    inline void set_boundary_free() { boundary_config.set_free(); }

    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;

    ExtForceConfig ext_force_config{};

    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
    }

    inline void
    add_ext_force_lense_thirring(Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
        ext_force_config.add_lense_thirring(central_mass, Racc, a_spin, dir_spin);
    }

    inline void add_ext_force_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {
        ext_force_config.add_shearing_box(Omega_0, eta, q);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool do_debug_dump = false;
    std::string debug_dump_filename = "";

    inline void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename){
        this->do_debug_dump = _do_debug_dump;
        this->debug_dump_filename = _debug_dump_filename;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////




    inline bool ghost_has_soundspeed() { return is_eos_locally_isothermal(); }

    inline bool has_field_uint() {
        // no barotropic for now
        return true;
    }

    inline bool has_field_alphaAV() { return artif_viscosity.has_alphaAV_field(); }

    inline bool has_field_divv() { return artif_viscosity.has_alphaAV_field(); }
    inline bool has_field_dtdivv() { return artif_viscosity.has_dtdivv_field(); }
    inline bool has_field_curlv() { return artif_viscosity.has_curlv_field() && (dim == 3); }

    inline bool has_field_soundspeed() {
        return artif_viscosity.has_field_soundspeed() || is_eos_locally_isothermal();
    }

    inline void print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- SPH Solver configuration -----");

        logger::raw_ln("units : ");
        if (unit_sys) {
            logger::raw_ln("unit_length      :", unit_sys->m_inv);
            logger::raw_ln("unit_mass        :", unit_sys->kg_inv);
            logger::raw_ln("unit_current     :", unit_sys->A_inv);
            logger::raw_ln("unit_temperature :", unit_sys->K_inv);
            logger::raw_ln("unit_qte         :", unit_sys->mol_inv);
            logger::raw_ln("unit_lumint      :", unit_sys->cd_inv);
        } else {
            logger::raw_ln("not set");
        }

        logger::raw_ln("part mass", gpart_mass, "( can be changed using .set_part_mass() )");
        logger::raw_ln("cfl force", cfl_force);
        logger::raw_ln("cfl courant", cfl_cour);

        artif_viscosity.print_status();
        eos_config.print_status();
        boundary_config.print_status();

        logger::raw_ln("------------------------------------");
    }

};
