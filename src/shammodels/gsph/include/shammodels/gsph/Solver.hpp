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
 * @file Solver.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief GSPH Solver class
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 *
 * This implementation follows:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shambase/exception.hpp"
#include "SolverConfig.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/SolverLog.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <memory>
#include <stdexcept>
#include <variant>

namespace shammodels::gsph {

    struct TimestepLog {
        i32 rank;
        f64 rate;
        u64 npart;
        f64 tcompute;

        inline f64 rate_sum() { return shamalgs::collective::allreduce_sum(rate); }
        inline u64 npart_sum() { return shamalgs::collective::allreduce_sum(npart); }
        inline f64 tcompute_max() { return shamalgs::collective::allreduce_max(tcompute); }
    };

    /**
     * @brief The GSPH Solver class
     *
     * Implements the Godunov SPH method using Riemann solvers at particle
     * interfaces instead of artificial viscosity.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4, M6, C2, C4, C6)
     */
    template<class Tvec, template<class> class SPHKernel>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;

        using u_morton = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        SolverStorage<Tvec, u_morton> storage{};

        Config solver_config;
        sph::SolverLog solve_logs;

        inline void init_required_fields() { solver_config.set_layout(context.get_pdl_write()); }

        // Serial patch tree control
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { storage.serial_patch_tree.reset(); }

        // Ghost handling - reuse SPH ghost handler
        using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        void gen_ghost_handler(Tscal time_val);
        inline void reset_ghost_handler() { storage.ghost_handler.reset(); }

        void build_ghost_cache();
        void clear_ghost_cache();

        void merge_position_ghost();

        // Tree operations
        using RTree = typename Config::RTree;
        void build_merged_pos_trees();
        void clear_merged_pos_trees();

        void compute_presteps_rint();
        void reset_presteps_rint();

        void start_neighbors_cache();
        void reset_neighbors_cache();

        void gsph_prestep(Tscal time_val, Tscal dt);

        void apply_position_boundary(Tscal time_val);

        void do_predictor_leapfrog(Tscal dt);

        void init_ghost_layout();

        void communicate_merge_ghosts_fields();
        void reset_merge_ghosts_fields();

        void compute_omega();
        void compute_eos_fields();
        void reset_eos_fields();

        /**
         * @brief Compute gradients for MUSCL reconstruction
         *
         * Computes density, pressure, and velocity gradients for each particle
         * using SPH kernel gradient summation. Only called when MUSCL reconstruction
         * is enabled (reconstruct_config.is_muscl() == true).
         *
         * Reference: Cha & Whitworth (2003)
         */
        void compute_gradients();

        void prepare_corrector();

        /**
         * @brief Update derivatives using GSPH Riemann solver
         *
         * This is the core GSPH step: for each particle pair, solve
         * the 1D Riemann problem and compute forces from the interface
         * pressure p*.
         */
        void update_derivs();

        /**
         * @brief Compute CFL timestep constraint
         *
         * Computes timestep from:
         * - Courant condition: dt_cour = C_cour * h / vsig
         * - Force condition: dt_force = C_force * sqrt(h / |a|)
         *
         * @return Minimum CFL timestep across all particles
         */
        Tscal compute_dt_cfl();

        bool apply_corrector(Tscal dt, u64 Npart_all);

        // =========================================================================
        // SR-GSPH integration methods (Special Relativistic mode)
        // =========================================================================

        /**
         * @brief Initialize SR conserved variables from primitives (prim2cons)
         *
         * Called on first timestep when SR is enabled. Converts (v, P, ρ) to (S, e).
         * Sets storage.sr_initialized = true after completion.
         */
        void sr_init_conserved();

        /**
         * @brief Initialize SR fields (S_momentum, e_energy, dS_momentum, de_energy)
         *
         * Must be called before SR integration can proceed.
         */
        void sr_init_fields();

        /**
         * @brief SR predictor step: advance S and e by half timestep
         *
         * S += dS * dt/2
         * e += de * dt/2
         * Then runs cons2prim to recover v for position integration.
         * Position: x += v * dt
         */
        void sr_do_predictor(Tscal dt);

        /**
         * @brief SR corrector step: advance S and e by half timestep
         *
         * S += dS * dt/2
         * e += de * dt/2
         * Then runs cons2prim to recover v, P, ρ for next step.
         */
        void sr_apply_corrector(Tscal dt);

        /**
         * @brief Convert conserved to primitive variables (cons2prim)
         *
         * Uses Newton-Raphson to solve for Lorentz factor, then recovers v, P, ρ.
         * Updates vxyz, pressure, density in particle data.
         */
        void sr_cons2prim();

        void update_sync_load_values();

        Solver(ShamrockCtx &context) : context(context) {}

        void init_solver_graph();

        void vtk_do_dump(std::string filename, bool add_patch_world_id);

        inline void print_timestep_logs() {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "GSPH", "iteration since start :", solve_logs.get_iteration_count());
                logger::info_ln(
                    "GSPH", "time since start :", shambase::details::get_wtime(), "(s)");
            }
        }

        TimestepLog evolve_once();

        Tscal evolve_once_time_expl(Tscal t_current, Tscal dt_input) {
            solver_config.set_time(t_current);
            solver_config.set_next_dt(dt_input);
            evolve_once();
            return solver_config.get_dt();
        }

        inline bool evolve_until(Tscal target_time, i32 niter_max = -1) {
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
                    logger::info_ln("GSPH", "stopping evolve until because of niter =", iter_count);
                    return false;
                }
            }

            print_timestep_logs();

            return true;
        }
    };

} // namespace shammodels::gsph
