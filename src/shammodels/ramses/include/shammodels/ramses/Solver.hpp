// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Solver.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
namespace shammodels::basegodunov {

    enum RiemmanSolverMode { Rusanov = 0, HLL = 1, HLLC = 2 };

    enum SlopeMode {
        None        = 0,
        VanLeer_f   = 1,
        VanLeer_std = 2,
        VanLeer_sym = 3,
        Minmod      = 4,
    };

    enum DustRiemannSolverMode {
        NoDust = 0,
        DHLL   = 1, // Dust HLL . This is merely the HLL solver for dust. It's then a Rusanov like
        HB     = 2 // Huang and Bai. Pressureless Riemann solver by Huang and Bai (2022) in Athena++
    };

    enum DragSolverMode {
        NoDrag = 0,
        IRK1   = 1, // Implicit RK1
        IRK2   = 2, // Implicit RK2
        EXPO   = 3  // Matrix exponential
    };

    /**
     * @brief alphas is the dust collision rate (the inverse of the stopping time)
     */
    struct DragConfig {
        DragSolverMode drag_solver_config = NoDrag;
        std::vector<f32> alphas;
        bool enable_frictional_heating
            = false; // 0 to turn off and 1 when all dissipation is deposited to the gas
    };

    struct DustConfig {
        DustRiemannSolverMode dust_riemann_config = NoDust;
        u32 ndust                                 = 0;

        inline bool is_dust_on() {
            if (dust_riemann_config != NoDust) {

                if (ndust == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "Dust is on with ndust == 0");
                }
                return true;
            }
            return false;
        }
    };

    template<class Tvec>
    struct SolverStatusVar {

        /// The type of the scalar used to represent the quantities
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal time = 0; ///< Current time
        Tscal dt   = 0; ///< Current time step
    };

    template<class Tvec, class TgridVec>
    struct AMRMode {

        using Tscal = shambase::VecComponent<Tvec>;

        struct None {};
        struct DensityBased {
            Tscal crit_mass;
        };

        using mode = std::variant<None, DensityBased>;

        mode config = None{};

        void set_refine_none() { config = None{}; }
        void set_refine_density_based(Tscal crit_mass) { config = DensityBased{crit_mass}; }
    };

    template<class Tvec, class TgridVec>
    struct SolverConfig {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal eos_gamma = 5. / 3.;

        Tscal grid_coord_to_pos_fact = 1;

        static constexpr u32 NsideBlockPow = 1;
        using AMRBlock                     = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

        inline void set_eos_gamma(Tscal gamma) { eos_gamma = gamma; }

        RiemmanSolverMode riemman_config  = HLL;
        SlopeMode slope_config            = VanLeer_sym;
        bool face_half_time_interpolation = true;

        //////////////////////////////////////////////////////////////////////////////////////////////
        // Dust config
        //////////////////////////////////////////////////////////////////////////////////////////////

        DustConfig dust_config{};
        DragConfig drag_config{};

        inline bool is_dust_on() { return dust_config.is_dust_on(); }
        // get alpha values from user
        // alphas is the dust collision rate (the inverse of the stopping time)
        inline void set_alphas_static(f32 alpha_values) {
            StackEntry stack_lock{};
            drag_config.alphas.push_back(alpha_values);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////
        // Dust config (END)
        //////////////////////////////////////////////////////////////////////////////////////////////

        /// AMR refinement mode
        AMRMode<Tvec, TgridVec> amr_mode = {};

        //////////////////////////////////////////////////////////////////////////////////////////////
        // Units Config
        //////////////////////////////////////////////////////////////////////////////////////////////

        /// The unit system of the simulation
        std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

        /// Set the unit system of the simulation
        inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }
        //////////////////////////////////////////////////////////////////////////////////////////////
        // Units Config (END)
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
        inline void set_next_dt(Tscal dt) { time_state.dt = dt; }
        /// Get the current time
        inline Tscal get_time() { return time_state.time; }
        /// Get the time step for the next iteration
        inline Tscal get_dt() { return time_state.dt; }

        Tscal Csafe = 0.9;
        //////////////////////////////////////////////////////////////////////////////////////////////
        // Solver status variables (END)
        //////////////////////////////////////////////////////////////////////////////////////////////
    };

    template<class Tvec, class TgridVec>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u64;
        using Config   = SolverConfig<Tvec, TgridVec>;

        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        SolverStorage<Tvec, TgridVec, u_morton> storage{};

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
            context.pdata_layout_add_field<Tscal>("rho", AMRBlock::block_size);
            context.pdata_layout_add_field<Tvec>("rhovel", AMRBlock::block_size);
            context.pdata_layout_add_field<Tscal>("rhoetot", AMRBlock::block_size);

            if (solver_config.is_dust_on()) {
                u32 ndust = solver_config.dust_config.ndust;

                context.pdata_layout_add_field<Tscal>("rho_dust", (ndust * AMRBlock::block_size));
                context.pdata_layout_add_field<Tvec>("rhovel_dust", (ndust * AMRBlock::block_size));
            }
        }

        Solver(ShamrockCtx &context) : context(context) {}

        void do_debug_vtk_dump(std::string filename);

        inline void print_timestep_logs() {
            if (shamcomm::world_rank() == 0) {
                // logger::info_ln("Godunov", "iteration since start :",
                // solve_logs.get_iteration_count());
                logger::info_ln(
                    "Godunov", "time since start :", shambase::details::get_wtime(), "(s)");
            }
        }

        void evolve_once();

        inline Tscal evolve_once_time_expl(Tscal t_current, Tscal dt_input) {
            solver_config.set_time(t_current);
            solver_config.set_next_dt(dt_input);
            evolve_once();
            return solver_config.get_dt();
        }

        inline bool evolve_until(Tscal target_time, i32 niter_max) {
            auto step = [&]() {
                Tscal dt = solver_config.get_dt();
                Tscal t  = solver_config.get_time();

                if (t > target_time) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "the target time is higher than the current time");
                }

                if (t + dt > target_time) {
                    solver_config.set_next_dt(target_time - t);
                }
                evolve_once();
            };

            i32 iter_count = 0;

            while (solver_config.get_time() < target_time) {
                step();
                iter_count++;

                if ((iter_count >= niter_max) && (niter_max != -1)) {
                    logger::info_ln("SPH", "stopping evolve until because of niter =", iter_count);
                    return false;
                }
            }

            print_timestep_logs();

            return true;
        }
    };

    /**
     * @brief Serialize a SolverConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The SolverConfig to serialize
     */
    template<class Tvec, class TgridVec>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p) {
        using T = SolverConfig<Tvec, TgridVec>;

        nlohmann::json junit;

        j = nlohmann::json{
            {"RiemmanSolverMode", p.riemman_config},
            {"SlopeMode", p.slope_config},
            {"DustRiemannSolverMode", p.Csafe},
            {"unit_sys", junit},
            {"time_state", p.time_state}};
    }

    namespace shamunits {

        /**
         * @brief Converts a UnitSystem object to a JSON object.
         *
         * @param j The JSON object to be populated.
         * @param p The UnitSystem object to be converted.
         */
        template<class Tscal>
        inline void to_json(nlohmann::json &j, const ::shamunits::UnitSystem<Tscal> &p) {
            j = nlohmann::json{
                {"unit_time", p.s_inv},
                {"unit_length", p.m_inv},
                {"unit_mass", p.kg_inv},
                {"unit_current", p.A_inv},
                {"unit_temperature", p.K_inv},
                {"unit_qte", p.mol_inv},
                {"unit_lumint", p.cd_inv}};
        }

        template<class Tvec>
        inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
            j = nlohmann::json{{"time", p.time}, {"dt", p.dt}};
        }

        template<class Tvec>
        inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
            using Tscal = typename SolverStatusVar<Tvec>::Tscal;
            j.at("time").get_to<Tscal>(p.time);
            j.at("dt").get_to<Tscal>(p.dt);
        }

    } // namespace shamunits

    template<class T>
    inline void to_json_optional(nlohmann::json &j, const std::optional<T> &p) {
        if (p) {
            j = *p;
        } else {
            j = {};
        }
    }

    /**
     * @brief Deserializes an optional value from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The optional value to populate.
     *
     * @return None
     *
     * @throws std::bad_optional_access if j is not a valid JSON object
     */
    template<class T>
    inline void from_json_optional(const nlohmann::json &j, std::optional<T> &p) {
        if (j.is_null()) {
            p = std::nullopt;
        } else {
            p = j.get<T>();
        }
    }

} // namespace shammodels::basegodunov
