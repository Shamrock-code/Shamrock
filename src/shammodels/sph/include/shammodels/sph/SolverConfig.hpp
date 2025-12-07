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
 * @file SolverConfig.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/EOSConfig.hpp"
#include "shammodels/common/ExtForceConfig.hpp"
#include "shammodels/sph/config/MHDConfig.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/io/units_json.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/RadixTree.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <variant>
#include <vector>

namespace shammodels::sph {

    /**
     * @brief The configuration for a sph solver
     *
     * @tparam Tvec the type of the vector used to represent the particles
     * @tparam SPHKernel the type of the SPH kernel
     */
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    /**
     * @brief Solver status variables
     *
     * @tparam Tvec the type of the vector used to represent the particles
     */
    template<class Tvec>
    struct SolverStatusVar;

    /**
     * @brief The configuration for the CFL condition
     *
     * @tparam Tscal the type of the scalar used to represent the quantities
     */
    template<class Tscal>
    struct CFLConfig {

        /**
         * @brief The CFL condition for the courant factor
         */
        Tscal cfl_cour;

        /**
         * @brief The CFL condition for the force
         */
        Tscal cfl_force;

        /**
         * @brief The CFL multiplier stiffness
         */
        Tscal cfl_multiplier_stiffness = 2;

        /// eta sink to control the sink integrator
        Tscal eta_sink = 0.05;
    };

    template<class Tvec>
    struct ParticleKillingConfig {
        using Tscal = shambase::VecComponent<Tvec>;
        struct Sphere {
            Tvec center;
            Tscal radius;
        };

        using kill_t = std::variant<Sphere>;

        std::vector<kill_t> kill_list;

        inline void add_kill_sphere(const Tvec &center, Tscal radius) {
            kill_list.push_back(Sphere{center, radius});
        }
    };

    template<class Tscal>
    struct DustConfig {

        struct None {};

        struct MonofluidTVI {
            u32 ndust;
        };

        struct MonofluidComplete {
            u32 ndust;
        };

        /// Variant type to store the EOS configuration
        using Variant = std::variant<None, MonofluidTVI, MonofluidComplete>;

        Variant current_mode = None{};

        inline void set_none() { current_mode = None{}; }
        inline void set_monofluid_tvi(u32 nvar) { current_mode = MonofluidTVI{nvar}; }
        inline void set_monofluid_complete(u32 nvar) { current_mode = MonofluidComplete{nvar}; }

        inline bool has_epsilon_field() {
            return bool(std::get_if<MonofluidTVI>(&current_mode))
                   || bool(std::get_if<MonofluidComplete>(&current_mode));
        }

        inline bool has_deltav_field() {
            return bool(std::get_if<MonofluidComplete>(&current_mode));
        }

        inline u32 get_dust_nvar() {
            if (None *cfg = std::get_if<None>(&current_mode)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Querrying a dust nvar with no dust as config is ... discutable ...");
                return 0;
            } else if (MonofluidTVI *cfg = std::get_if<MonofluidTVI>(&current_mode)) {
                return cfg->ndust;
            } else if (MonofluidComplete *cfg = std::get_if<MonofluidComplete>(&current_mode)) {
                return cfg->ndust;
            } else {
                shambase::throw_unimplemented("How did you get here ???");
            }
            return 0;
        }

        inline void check_config() {
            bool is_not_none = bool(std::get_if<MonofluidTVI>(&current_mode))
                               || bool(std::get_if<MonofluidComplete>(&current_mode));
            if (is_not_none) {
                ON_RANK_0(
                    logger::warn_ln(
                        "SPH::config",
                        "Dust config != None is work in progress, use it at your own risk"));
            }
        }
    };

    struct SmoothingLengthConfig {
        struct DensityBased {};
        struct DensityBasedNeighLim {
            u32 max_neigh_count = 500;
        };

        using mode = std::variant<DensityBased, DensityBasedNeighLim>;

        mode config = DensityBased{};

        void set_density_based() { config = DensityBased{}; }
        void set_density_based_neigh_lim(u32 max_neigh_count) {
            config = DensityBasedNeighLim{max_neigh_count};
        }

        bool is_density_based_neigh_lim() const {
            return std::holds_alternative<DensityBasedNeighLim>(config);
        }
    };

    struct SelfGravConfig {

        struct MM {
            u32 order;
            f64 opening_angle;
            u32 reduction_level;
        };

        struct Direct {
            bool reference_mode = false;
        };

        struct None {};

        using mode = std::variant<MM, Direct, None>;

        mode config = None{};

        void set_none() { config = None{}; }
        void set_direct(bool reference_mode = false) { config = Direct{reference_mode}; }
        void set_mm(u32 mm_order, f64 opening_angle, u32 reduction_level) {
            config = MM{mm_order, opening_angle, reduction_level};
        }

        bool is_none() const { return std::holds_alternative<None>(config); }
        bool is_direct() const { return std::holds_alternative<Direct>(config); }

        bool is_mm() const { return std::holds_alternative<MM>(config); }

        bool is_sg_on() const { return !is_none(); }
        bool is_sg_off() const { return is_none(); }

        struct SofteningPlummer {
            f64 epsilon;
        };

        using mode_soft          = std::variant<SofteningPlummer>;
        mode_soft softening_mode = SofteningPlummer{1e-9};

        void set_softening_plummer(f64 epsilon) { softening_mode = SofteningPlummer{epsilon}; }
        void set_softening_none() { set_softening_plummer(0.); }

        bool is_softening_plummer() const {
            return std::holds_alternative<SofteningPlummer>(softening_mode);
        }
    };

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::SolverStatusVar {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time   = 0; ///< Current time
    Tscal dt_sph = 0; ///< Current time step

    Tscal cfl_multiplier = 1e-2; ///< Current cfl multiplier
};

template<class Tvec, template<class> class SPHKernel>
struct shammodels::sph::SolverConfig {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;
    /// The dimension of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    /// The type of the kernel used for the SPH interactions
    using Kernel = SPHKernel<Tscal>;
    /// The type of the Morton code for the tree
    using u_morton = u32;

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    /// The radius of the sph kernel
    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass;            ///< The mass of each gas particle
    CFLConfig<Tscal> cfl_config; ///< The configuration for the CFL condition

    bool track_particles_id = false;

    inline void set_particle_tracking(bool state) { track_particles_id = state; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// The unit system of the simulation
    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    /// Set the unit system of the simulation
    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }

    /// Retrieves the value of the constant G based on the unit system.
    inline Tscal get_constant_G() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }

    /// Retrieves the value of the constant c based on the unit system.
    inline Tscal get_constant_c() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.c();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.c();
        }
    }

    /// Retrieves the value of the constant mu_0 based on the unit system.
    inline Tscal get_constant_mu_0() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.mu_0();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.mu_0();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Particle killing config
    //////////////////////////////////////////////////////////////////////////////////////////////

    ParticleKillingConfig<Tvec> particle_killing;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Particle killing config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to SolverStatusVar type
    using SolverStatusVar = SolverStatusVar<Tvec>;

    /// The time sate of the simulation
    SolverStatusVar time_state;

    /// Set the current time
    inline void set_time(Tscal t) { time_state.time = t; }

    /// Set the time step for the next iteration
    inline void set_next_dt(Tscal dt) { time_state.dt_sph = dt; }

    /// Get the current time
    inline Tscal get_time() { return time_state.time; }

    /// Get the time step for the next iteration
    inline Tscal get_dt_sph() { return time_state.dt_sph; }

    /// Set the CFL multiplier for the time step
    inline void set_cfl_multipler(Tscal lambda) { time_state.cfl_multiplier = lambda; }

    /// Get the CFL multiplier for the time step
    inline Tscal get_cfl_multipler() { return time_state.cfl_multiplier; }

    /// Set the CFL multiplier for the stiffness
    inline void set_cfl_mult_stiffness(Tscal cstiff) {
        cfl_config.cfl_multiplier_stiffness = cstiff;
    }

    /// Get the CFL multiplier for the stiffness
    inline Tscal get_cfl_mult_stiffness() { return cfl_config.cfl_multiplier_stiffness; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using MHDConfig      = MHDConfig<Tvec>;
    MHDConfig mhd_config = {};

    /// disable MHD in the SPH solver
    inline void set_noMHD() {
        using Tmp = typename MHDConfig::None;
        mhd_config.set(Tmp{});
    }

    /// Enable the ideal MHD hydro solver
    inline void set_IdealMHD(typename MHDConfig::IdealMHD_constrained_hyper_para v) {
        mhd_config.set(v);
    }

    inline void set_NonIdealMHD(typename MHDConfig::NonIdealMHD v) { mhd_config.set(v); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using DustConfig       = DustConfig<Tscal>;
    DustConfig dust_config = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Self gravity config
    //////////////////////////////////////////////////////////////////////////////////////////////

    SelfGravConfig self_grav_config = SelfGravConfig{};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Self gravity config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;    ///< Reduction level to be used in the tree build
    bool use_two_stage_search = true; ///< Use two stage neighbors search (see shamrock paper)

    /// Setter for the tree reduction level
    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    /// Setter for the two stage search
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }

    bool show_neigh_stats = false;
    inline void set_show_neigh_stats(bool enable) { show_neigh_stats = enable; }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool combined_dtdiv_divcurlv_compute = false; ///< Use the combined dtdivv and divcurlv compute
    /// Factor applied to the smoothing length for neighbors search (and ghost zone size)
    /// @note This value must be larger or equal to htol_up_fine_cycle
    Tscal htol_up_coarse_cycle = 1.1;
    /// Maximum factor of the smoothing length evolution per subcycles
    Tscal htol_up_fine_cycle  = 1.1;
    Tscal epsilon_h           = 1e-6; ///< Convergence criteria for the smoothing length
    u32 h_iter_per_subcycles  = 50;   ///< Maximum number of iterations per subcycle
    u32 h_max_subcycles_count = 100;  ///< Maximum number of subcycles before solver crash

    SmoothingLengthConfig smoothing_length_config;

    inline void set_smoothing_length_density_based() {
        smoothing_length_config.set_density_based();
    }
    inline void set_smoothing_length_density_based_neigh_lim(u32 max_neigh_count) {
        smoothing_length_config.set_density_based_neigh_lim(max_neigh_count);
    }

    bool enable_particle_reordering = false;
    inline void set_enable_particle_reordering(bool enable) { enable_particle_reordering = enable; }
    u64 particle_reordering_step_freq = 1000;
    inline void set_particle_reordering_step_freq(u64 freq) {
        if (freq == 0) {
            shambase::throw_with_loc<std::invalid_argument>(
                "particle_reordering_step_freq cannot be zero");
        }
        particle_reordering_step_freq = freq;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to EOSConfig type
    using EOSConfig = shammodels::EOSConfig<Tvec>;

    /// EOS configuration
    EOSConfig eos_config;

    /// Check if the EOS is a locally isothermal equation of state
    inline bool is_eos_locally_isothermal() {
        using T = typename EOSConfig::LocallyIsothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is an adiabatic equation of state
    inline bool is_eos_adiabatic() {
        using T = typename EOSConfig::Adiabatic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is a polytropic equation of state
    inline bool is_eos_polytropic() {
        using T = typename EOSConfig::Polytropic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is an isothermal equation of state
    inline bool is_eos_isothermal() {
        using T = typename EOSConfig::Isothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is a Fermi equation of state
    inline bool is_eos_fermi() {
        using T = typename EOSConfig::Fermi;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /**
     * @brief Set the EOS configuration to an isothermal equation of state
     *
     * @param cs The isothermal index
     */
    inline void set_eos_isothermal(Tscal cs) { eos_config.set_isothermal(cs); }

    /**
     * @brief Set the EOS configuration to an adiabatic equation of state
     *
     * @param gamma The adiabatic index
     */
    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }

    /**
     * @brief Set the EOS configuration to an polytropic equation of state
     *
     * @param gamma The adiabatic index
     */
    inline void set_eos_polytropic(Tscal K, Tscal gamma) { eos_config.set_polytropic(K, gamma); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state
     */
    inline void set_eos_locally_isothermal() { eos_config.set_locally_isothermal(); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state from Lodato
     * Price 2007
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     */
    inline void set_eos_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
        eos_config.set_locally_isothermalLP07(cs0, q, r0);
    }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state fromFarris 2014
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     */
    inline void set_eos_locally_isothermalFA2014(Tscal h_over_r) {
        eos_config.set_locally_isothermalFA2014(h_over_r);
    }

    /**
     * @brief Set the EOS configuration to a Fermi equation of state
     *
     */
    inline void set_eos_fermi(Tscal mu_e) { eos_config.set_fermi(mu_e); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration for the Artificial Viscosity (AV)
     *
     * @details This struct contains the information needed to configure the Artificial Viscosity
     * in the SPH algorithm. It is a variant of two possible types of artificial viscosity:
     * - None: no AV
     * - Constant: AV with a constant value
     * - VaryingMM97: AV with a varying value, using the Monaghan & Gingold 1997 prescription
     * - VaryingCD10: AV with a varying value, using the Cullen & Dehnen 2010 prescription
     * - ConstantDisc: AV with a constant value, but only in the disc plane
     */
    using AVConfig = AVConfig<Tvec>;

    /// Configuration for the Artificial Viscosity (AV)
    AVConfig artif_viscosity;

    /**
     * @brief Set the artificial viscosity configuration to None
     */
    inline void set_artif_viscosity_None() {
        using Tmp = typename AVConfig::None;
        artif_viscosity.set(Tmp{});
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value
     *
     * @param v Constant value of the artificial viscosity
     */
    inline void set_artif_viscosity_Constant(typename AVConfig::Constant v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Monaghan & Gingold 1997
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingMM97(typename AVConfig::VaryingMM97 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Cullen & Dehnen 2010
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingCD10(typename AVConfig::VaryingCD10 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value in the disc plane.
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) {
        artif_viscosity.set(v);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration of the boundary conditions
     */
    using BCConfig = BCConfig<Tvec>;

    /**
     * @brief Boundary condition configuration
     *
     * See the documentation of the `BCConfig` struct for more informations.
     */
    BCConfig boundary_config;

    /**
     * @brief Set the boundary condition to free boundary
     */
    inline void set_boundary_free() { boundary_config.set_free(); }

    /**
     * @brief Set the boundary condition to periodic boundary
     */
    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    /**
     * @brief Set the boundary condition to shearing periodic boundary
     *
     * The particles are periodic in all directions, but with a shear in the direction
     * given by `shear_dir` and a period of `speed`.
     *
     * @param[in] shear_base The base of the scalar product to define the number of shearing
     * periodicity to be applied
     * @param[in] shear_dir The direction of the shear
     * @param[in] speed The speed of the shear
     */
    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief External force configuration
     *
     * This configuration is used to define the external forces that are applied to the
     * particles in the simulation.
     *
     * The external forces are defined by a variant of different types of forces
     * (i.e., point mass, Lense-Thirring, etc.). The user can add different types
     * of forces using the functions `add_ext_force_point_mass`, `add_ext_force_lense_thirring`,
     * etc.
     */
    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;

    /**
     * @brief External force configuration
     */
    ExtForceConfig ext_force_config{};

    /**
     * @brief Add a point mass external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     */
    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
    }

    /**
     * @brief Add a Lense-Thirring external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     * @param[in] a_spin The spin of the central object
     * @param[in] dir_spin The direction of the spin of the central object
     */
    inline void add_ext_force_lense_thirring(
        Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
        ext_force_config.add_lense_thirring(central_mass, Racc, a_spin, dir_spin);
    }

    /**
     * @brief Add a shearing box external force
     *
     * @param[in] Omega_0 The angular frequency of the shear
     * @param[in] eta The shear rate
     * @param[in] q The power-law index of the shear
     */
    inline void add_ext_force_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {
        ext_force_config.add_shearing_box(Omega_0, eta, q);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether to dump debug information to file
    bool do_debug_dump = false;

    /// @brief The filename to dump debug information in
    std::string debug_dump_filename = "";

    /// @brief Set whether to dump debug information to file
    ///
    /// @param[in] _do_debug_dump Whether to dump debug information to file
    /// @param[in] _debug_dump_filename The filename to dump debug information to
    inline void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename) {
        this->do_debug_dump       = _do_debug_dump;
        this->debug_dump_filename = _debug_dump_filename;
    }

    /// @brief Whether to add debug fields to the pdl.
    inline constexpr bool do_MHD_debug() { return false; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether the ghost cells have a sound speed (i.e. the eos is locally isothermal)
    inline bool ghost_has_soundspeed() { return is_eos_locally_isothermal(); }

    /// @brief Whether the solver has a field for the particle's uint
    ///
    /// @note for now, this is always true as
    inline bool has_field_uint() {
        // no barotropic for now
        return true;
    }

    /// @brief Whether the solver has a field for alpha AV
    inline bool has_field_alphaAV() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for divv
    inline bool has_field_divv() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for dt divv
    inline bool has_field_dtdivv() { return artif_viscosity.has_dtdivv_field(); }

    /// @brief Whether the solver has a field for curlv
    inline bool has_field_curlv() { return artif_viscosity.has_curlv_field() && (dim == 3); }

    /// @brief Whether the solver has a field for ax, ay, az in ghost cells
    inline bool has_axyz_in_ghost() { return has_field_dtdivv(); }

    /// @brief Whether the solver has a field for sound speed
    inline bool has_field_soundspeed() {
        return artif_viscosity.has_field_soundspeed() || is_eos_locally_isothermal();
    }

    /// @brief Whether the solver has a field for B_on_rho
    inline bool has_field_B_on_rho() { return mhd_config.has_B_field() && (dim == 3); }

    /// @brief Whether the solver has a field for psi_on_ch
    inline bool has_field_psi_on_ch() { return mhd_config.has_psi_field(); }

    /// @brief Whether the solver has a field for divB
    inline bool has_field_divB() { return mhd_config.has_divB_field(); }

    /// @brief Whether the solver has a field for curlB
    inline bool has_field_curlB() { return mhd_config.has_curlB_field() && (dim == 3); }

    /// @brief Whether the solver has a field for dt divB
    inline bool has_field_dtdivB() { return mhd_config.has_dtdivB_field(); }

    /// Print the current status of the solver config
    inline void print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- SPH Solver configuration -----");
        logger::raw_ln(nlohmann::json{*this}.dump(4));
        logger::raw_ln("------------------------------------");
    }

    inline void check_config() {
        dust_config.check_config();

        if (track_particles_id && false /*particle injection when added*/) {
            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "particle injection is not yet compatible with particle id tracking");
            }
        }

        if (track_particles_id) {
            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>("Particle tracking is experimental");
            }
        }

        if (!self_grav_config.is_none()) {
            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Self gravity is experimental, please enable experimental features to use it");
            }
        }
    }

    void set_layout(shamrock::patch::PatchDataLayerLayout &pdl);
    void set_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout);
};

namespace shammodels::sph {

    /**
     * @brief Converts a CFLConfig object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The CFLConfig object to be converted.
     */
    template<class Tscal>
    inline void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
            {"cfl_multiplier_stiffness", p.cfl_multiplier_stiffness},
            {"eta_sink", p.eta_sink}};
    }

    /**
     * @brief Deserializes a CFLConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The CFLConfig object to populate.
     */
    template<class Tscal>
    inline void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to<Tscal>(p.cfl_cour);
        j.at("cfl_force").get_to<Tscal>(p.cfl_force);
        j.at("cfl_multiplier_stiffness").get_to<Tscal>(p.cfl_multiplier_stiffness);

        if (j.contains("eta_sink")) {
            j.at("eta_sink").get_to<Tscal>(p.eta_sink);
        } else {
            // Already set to default value
            ON_RANK_0(shamlog_warn_ln(
                "SPHConfig", "eta_sink not found when deserializing, defaulting to", p.eta_sink));
        }
    }

    /**
     * @brief Converts a SolverStatusVar object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The SolverStatusVar object to be converted.
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
        j = nlohmann::json{
            {"time", p.time}, {"dt_sph", p.dt_sph}, {"cfl_multiplier", p.cfl_multiplier}};
    }

    /**
     * @brief Deserializes a SolverStatusVar object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverStatusVar object to populate.
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
        using Tscal = typename SolverStatusVar<Tvec>::Tscal;
        j.at("time").get_to<Tscal>(p.time);
        j.at("dt_sph").get_to<Tscal>(p.dt_sph);
        j.at("cfl_multiplier").get_to<Tscal>(p.cfl_multiplier);
    }

    // JSON serialization for ParticleKillingConfig
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ParticleKillingConfig<Tvec> &p) {
        j = nlohmann::json::array();
        for (const auto &kill : p.kill_list) {
            if (std::holds_alternative<typename ParticleKillingConfig<Tvec>::Sphere>(kill)) {
                const auto &sphere = std::get<typename ParticleKillingConfig<Tvec>::Sphere>(kill);
                j.push_back(
                    {{"type", "sphere"}, {"center", sphere.center}, {"radius", sphere.radius}});
            }
            // If more types are added to kill_t, handle them here
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ParticleKillingConfig<Tvec> &p) {
        p.kill_list.clear();
        for (const auto &item : j) {
            std::string type = item.at("type").get<std::string>();
            if (type == "sphere") {
                typename ParticleKillingConfig<Tvec>::Sphere sphere;
                item.at("center").get_to(sphere.center);
                item.at("radius").get_to(sphere.radius);
                p.kill_list.push_back(sphere);
            }
            // If more types are added to kill_t, handle them here
        }
    }

    // JSON serialization for SmoothingLengthConfig
    inline void to_json(nlohmann::json &j, const SmoothingLengthConfig &p) {
        if (const SmoothingLengthConfig::DensityBased *conf
            = std::get_if<SmoothingLengthConfig::DensityBased>(&p.config)) {
            j = {
                {"type", "density_based"},
            };

        } else if (
            const SmoothingLengthConfig::DensityBasedNeighLim *conf
            = std::get_if<SmoothingLengthConfig::DensityBasedNeighLim>(&p.config)) {

            j = {
                {"type", "density_based_neigh_lim"},
                {"max_neigh_count", conf->max_neigh_count},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    inline void from_json(const nlohmann::json &j, SmoothingLengthConfig &p) {
        if (j.at("type").get<std::string>() == "density_based") {
            p.config = SmoothingLengthConfig::DensityBased{};
        } else if (j.at("type").get<std::string>() == "density_based_neigh_lim") {
            p.config
                = SmoothingLengthConfig::DensityBasedNeighLim{j.at("max_neigh_count").get<u32>()};
        } else {
            shambase::throw_unimplemented();
        }
    }

    /// JSON serialization for SelfGravConfig
    inline void to_json(nlohmann::json &j, const SelfGravConfig &p) {
        if (const SelfGravConfig::MM *conf = std::get_if<SelfGravConfig::MM>(&p.config)) {
            j = {
                {"type", "mm"},
                {"order", conf->order},
                {"opening_angle", conf->opening_angle},
                {"reduction_level", conf->reduction_level},
            };
        } else if (
            const SelfGravConfig::Direct *conf = std::get_if<SelfGravConfig::Direct>(&p.config)) {
            j = {
                {"type", "direct"},
                {"reference_mode", conf->reference_mode},
            };
        } else if (
            const SelfGravConfig::None *conf = std::get_if<SelfGravConfig::None>(&p.config)) {
            j = {
                {"type", "none"},
            };
        }

        if (const SelfGravConfig::SofteningPlummer *conf
            = std::get_if<SelfGravConfig::SofteningPlummer>(&p.softening_mode)) {
            j["softening_mode"]   = "plummer";
            j["softening_length"] = conf->epsilon;
        } else {
            shambase::throw_unimplemented();
        }
    }

    /// JSON deserialization for SelfGravConfig
    inline void from_json(const nlohmann::json &j, SelfGravConfig &p) {
        if (j.at("type").get<std::string>() == "mm") {
            p.config = SelfGravConfig::MM{
                j.at("order").get<u32>(),
                j.at("opening_angle").get<f64>(),
                j.at("reduction_level").get<u32>()};
        } else if (j.at("type").get<std::string>() == "direct") {
            p.config = SelfGravConfig::Direct{j.at("reference_mode").get<bool>()};
        } else if (j.at("type").get<std::string>() == "none") {
            p.config = SelfGravConfig::None{};
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid self gravity type: " + j.at("type").get<std::string>());
        }

        if (j.contains("softening_mode")) {
            std::string softening_mode = j.at("softening_mode").get<std::string>();
            if (softening_mode == "plummer") {
                p.softening_mode
                    = SelfGravConfig::SofteningPlummer{j.at("softening_length").get<f64>()};
            } else {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "Invalid softening mode: " + softening_mode);
            }
        }
    }

    /**
     * @brief Serializes a SolverConfig object to a JSON object.
     *
     * @param j The JSON object to serialize to.
     * @param p The SolverConfig object to serialize.
     */
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
            // mhd config
            {"mhd_config", p.mhd_config},
            // self gravity config
            {"self_grav_config", p.self_grav_config},
            // tree config
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"show_neigh_stats", p.show_neigh_stats},
            // solver behavior config
            {"combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute},
            {"htol_up_coarse_cycle", p.htol_up_coarse_cycle},
            {"htol_up_fine_cycle", p.htol_up_fine_cycle},
            {"epsilon_h", p.epsilon_h},
            {"smoothing_length_config", p.smoothing_length_config},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},

            {"enable_particle_reordering", p.enable_particle_reordering},
            {"particle_reordering_step_freq", p.particle_reordering_step_freq},

            {"eos_config", p.eos_config},

            {"artif_viscosity", p.artif_viscosity},
            {"boundary_config", p.boundary_config},
            {"ext_force_config", p.ext_force_config},

            {"do_debug_dump", p.do_debug_dump},
            {"debug_dump_filename", p.debug_dump_filename},
            // particle killing config
            {"particle_killing", p.particle_killing},
        };
    }

    /**
     * @brief Deserializes a SolverConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverConfig object to populate.
     */
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

        // mhd config
        try {
            j.at("mhd_config").get_to(p.mhd_config);
        } catch (const nlohmann::json::out_of_range &e) {
            logger::warn_ln(
                "SPHConfig", "mhd_config not found when deserializing, defaulting to None");
            p.mhd_config.set(typename T::MHDConfig::None{});
        }

        // self gravity config
        if (j.contains("self_grav_config")) {
            j.at("self_grav_config").get_to(p.self_grav_config);
        } else {
            logger::warn_ln(
                "SPHConfig",
                "self_grav_config not found when deserializing, defaulting to ",
                nlohmann::json{p.self_grav_config}.dump(4));
        }

        j.at("tree_reduction_level").get_to(p.tree_reduction_level);
        j.at("use_two_stage_search").get_to(p.use_two_stage_search);

        if (j.contains("show_neigh_stats")) {
            j.at("show_neigh_stats").get_to(p.show_neigh_stats);
        } else {
            // Already set to default value
            ON_RANK_0(shamlog_warn_ln(
                "SPHConfig",
                "show_neigh_stats not found when deserializing, defaulting to ",
                p.show_neigh_stats));
        }

        j.at("combined_dtdiv_divcurlv_compute").get_to(p.combined_dtdiv_divcurlv_compute);

        // Try new names first, fall back to old names for backward compatibility
        if (j.contains("htol_up_coarse_cycle")) {
            j.at("htol_up_coarse_cycle").get_to(p.htol_up_coarse_cycle);
        } else {
            j.at("htol_up_tol").get_to(p.htol_up_coarse_cycle);
        }

        if (j.contains("htol_up_fine_cycle")) {
            j.at("htol_up_fine_cycle").get_to(p.htol_up_fine_cycle);
        } else {
            j.at("htol_up_iter").get_to(p.htol_up_fine_cycle);
        }

        j.at("epsilon_h").get_to(p.epsilon_h);

        if (j.contains("smoothing_length_config")) {
            j.at("smoothing_length_config").get_to(p.smoothing_length_config);
        } else {
            logger::warn_ln(
                "SPHConfig",
                "smoothing_length_config not found when deserializing, defaulting to ",
                nlohmann::json{p.smoothing_length_config}.dump(4));
        }

        j.at("h_iter_per_subcycles").get_to(p.h_iter_per_subcycles);
        j.at("h_max_subcycles_count").get_to(p.h_max_subcycles_count);

        if (j.contains("enable_particle_reordering")) {
            j.at("enable_particle_reordering").get_to(p.enable_particle_reordering);
        } else {
            logger::warn_ln(
                "SPHConfig",
                "enable_particle_reordering not found when deserializing, defaulting to ",
                p.enable_particle_reordering);
        }

        if (j.contains("particle_reordering_step_freq")) {
            j.at("particle_reordering_step_freq").get_to(p.particle_reordering_step_freq);
        } else {
            logger::warn_ln(
                "SPHConfig",
                "particle_reordering_step_freq not found when deserializing, defaulting to ",
                p.particle_reordering_step_freq);
        }

        j.at("eos_config").get_to(p.eos_config);
        j.at("artif_viscosity").get_to(p.artif_viscosity);
        j.at("boundary_config").get_to(p.boundary_config);
        j.at("ext_force_config").get_to(p.ext_force_config);

        j.at("do_debug_dump").get_to(p.do_debug_dump);
        j.at("debug_dump_filename").get_to(p.debug_dump_filename);

        // particle killing config
        try {
            j.at("particle_killing").get_to(p.particle_killing);
        } catch (const nlohmann::json::out_of_range &e) {
            logger::warn_ln(
                "SPHConfig", "particle_killing not found when deserializing, defaulting to None");
            p.particle_killing.kill_list = {};
        }
    }

} // namespace shammodels::sph
