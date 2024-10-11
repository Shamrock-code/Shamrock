// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
#include "shammodels/amr/AMRBlock.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
namespace shammodels::basegodunov {

    enum RiemmanSolverMode { Rusanov = 0, HLL = 1 };

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
        DustConfig dust_config{};

        inline bool is_dust_on() { return dust_config.is_dust_on(); }

        Tscal Csafe = 0.9;

        /// AMR refinement mode
        AMRMode<Tvec, TgridVec> amr_mode = {};

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

} // namespace shammodels::basegodunov
