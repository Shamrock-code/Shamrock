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

#include "shambase/exception.hpp"
#include "config/AVConfig.hpp"
#include "config/BCConfig.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/EOSConfig.hpp"
#include "shammodels/ExtForceConfig.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <variant>

namespace shammodels::sph {
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    template<class Tvec>
    struct SolverStatusVar;

    template<class Tscal>
    struct CFLConfig {

        Tscal cfl_cour;
        Tscal cfl_force;
        Tscal cfl_multiplier_stiffness = 2;
    };

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
    CFLConfig<Tscal> cfl_config;

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

    inline void set_cfl_multipler(Tscal lambda) { time_state.cfl_multiplier = lambda; }
    inline Tscal get_cfl_multipler() { return time_state.cfl_multiplier; }

    inline void set_cfl_mult_stiffness(Tscal cstiff) {
        cfl_config.cfl_multiplier_stiffness = cstiff;
    }
    inline Tscal get_cfl_mult_stiffness() { return cfl_config.cfl_multiplier_stiffness; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;
    bool use_two_stage_search = true;
    u64 max_neigh_cache_size  = 10e9;

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
    Tscal htol_up_tol                    = 1.1;
    Tscal htol_up_iter                   = 1.1;
    Tscal epsilon_h                      = 1e-6;
    u32 h_iter_per_subcycles             = 50;
    u32 h_max_subcycles_count            = 100;

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

    inline bool is_eos_adiabatic() {
        using T = typename EOSConfig::Adiabatic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }
    inline void set_eos_locally_isothermal() { eos_config.set_locally_isothermal(); }
    inline void set_eos_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
        eos_config.set_locally_isothermalLP07(cs0, q, r0);
    }

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

    using BCConfig = BCConfig<Tvec>;

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
    // Debug dump config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool do_debug_dump              = false;
    std::string debug_dump_filename = "";

    inline void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename) {
        this->do_debug_dump       = _do_debug_dump;
        this->debug_dump_filename = _debug_dump_filename;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config (END)
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

    inline bool has_axyz_in_ghost() { return has_field_dtdivv(); }

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
        logger::raw_ln("cfl force", cfl_config.cfl_force);
        logger::raw_ln("cfl courant", cfl_config.cfl_cour);

        artif_viscosity.print_status();
        eos_config.print_status();
        boundary_config.print_status();

        logger::raw_ln("------------------------------------");
    }
};

namespace shamunits {
    template<class Tscal>
    inline void to_json(nlohmann::json &j, const UnitSystem<Tscal> &p) {
        j = nlohmann::json{
            {"unit_time", p.s_inv},
            {"unit_length", p.m_inv},
            {"unit_mass", p.kg_inv},
            {"unit_current", p.A_inv},
            {"unit_temperature", p.K_inv},
            {"unit_qte", p.mol_inv},
            {"unit_lumint", p.cd_inv}};
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, UnitSystem<Tscal> &p) {
        p = UnitSystem<Tscal>(
            j.at("unit_time").get<Tscal>(),
            j.at("unit_length").get<Tscal>(),
            j.at("unit_mass").get<Tscal>(),
            j.at("unit_current").get<Tscal>(),
            j.at("unit_temperature").get<Tscal>(),
            j.at("unit_qte").get<Tscal>(),
            j.at("unit_lumint").get<Tscal>());
    }
} // namespace shamunits

namespace shammodels::sph {

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
            {"cfl_multiplier_stiffness", p.cfl_multiplier_stiffness}};
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to<Tscal>(p.cfl_cour);
        j.at("cfl_force").get_to<Tscal>(p.cfl_force);
        j.at("cfl_multiplier_stiffness").get_to<Tscal>(p.cfl_multiplier_stiffness);
    }

    template<class T>
    inline void to_json_optional(nlohmann::json &j, const std::optional<T> &p) {
        if (p) {
            j = *p;
        } else {
            j = {};
        }
    }

    template<class T>
    inline void from_json_optional(const nlohmann::json &j, std::optional<T> &p) {
        if (j.is_null()) {
            p = std::nullopt;
        } else {
            p = j.get<T>();
        }
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
        j = nlohmann::json{
            {"time", p.time}, {"dt_sph", p.dt_sph}, {"cfl_multiplier", p.cfl_multiplier}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
        using Tscal = typename SolverStatusVar<Tvec>::Tscal;
        j.at("time").get_to<Tscal>(p.time);
        j.at("dt_sph").get_to<Tscal>(p.dt_sph);
        j.at("cfl_multiplier").get_to<Tscal>(p.cfl_multiplier);
    }

    template<class Tvec, template<class> class SPHKernel>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        nlohmann::json junit;
        to_json_optional(junit, p.unit_sys);

        j = nlohmann::json{
            // used for type checking
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            // actual data stored in the json
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", junit},
            {"time_state", p.time_state},
            // tree config
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"max_neigh_cache_size", p.max_neigh_cache_size},
            // solver behavior config
            {"combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute},
            {"htol_up_tol", p.htol_up_tol},
            {"htol_up_iter", p.htol_up_iter},
            {"epsilon_h", p.epsilon_h},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},

            {"eos_config", p.eos_config},

            {"artif_viscosity", p.artif_viscosity},
            {"boundary_config", p.boundary_config},
            {"ext_force_config", p.ext_force_config},

            {"do_debug_dump", p.do_debug_dump},
            {"debug_dump_filename", p.debug_dump_filename},
        };
    }

    template<class Tvec, template<class> class SPHKernel>
    inline void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        // type checking
        std::string kernel_id = j.at("kernel_id").get<std::string>();

        if (kernel_id != shambase::get_type_name<Tkernel>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid type to deserialize, wanted " + shambase::get_type_name<Tkernel>()
                + " but got " + kernel_id);
        }

        std::string type_id = j.at("type_id").get<std::string>();

        if (type_id != shambase::get_type_name<Tvec>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                + " but got " + type_id);
        }

        // actual data stored in the json
        j.at("gpart_mass").get_to(p.gpart_mass);
        j.at("cfl_config").get_to(p.cfl_config);

        from_json_optional(j.at("unit_sys"), p.unit_sys);

        j.at("time_state").get_to(p.time_state);

        j.at("tree_reduction_level").get_to(p.tree_reduction_level);
        j.at("use_two_stage_search").get_to(p.use_two_stage_search);
        j.at("max_neigh_cache_size").get_to(p.max_neigh_cache_size);

        j.at("combined_dtdiv_divcurlv_compute").get_to(p.combined_dtdiv_divcurlv_compute);
        j.at("htol_up_tol").get_to(p.htol_up_tol);
        j.at("htol_up_iter").get_to(p.htol_up_iter);
        j.at("epsilon_h").get_to(p.epsilon_h);
        j.at("h_iter_per_subcycles").get_to(p.h_iter_per_subcycles);
        j.at("h_max_subcycles_count").get_to(p.h_max_subcycles_count);

        j.at("eos_config").get_to(p.eos_config);
        j.at("artif_viscosity").get_to(p.artif_viscosity);
        j.at("boundary_config").get_to(p.boundary_config);
        j.at("ext_force_config").get_to(p.ext_force_config);

        j.at("do_debug_dump").get_to(p.do_debug_dump);
        j.at("debug_dump_filename").get_to(p.debug_dump_filename);
    }

} // namespace shammodels::sph
