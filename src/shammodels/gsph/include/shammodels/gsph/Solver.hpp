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
#include "shammodels/gsph/core/PhysicsMode.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shammodels/gsph/physics/mhd/MHDConfig.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianConfig.hpp"
#include "shammodels/gsph/physics/sr/SRConfig.hpp"
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
     * Physics mode is selected via set_physics_*() methods. Each mode owns
     * its own configuration - no central PhysicsConfig variant.
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

        /// Physics mode - handles physics-specific time stepping
        /// Each mode owns its own config (NewtonianConfig, SRConfig, MHDConfig)
        std::unique_ptr<core::PhysicsMode<Tvec, SPHKernel>> physics_mode;

        inline void init_required_fields() {
            solver_config.set_layout(context.get_pdl_write());
            // Physics mode extends layout with physics-specific fields
            if (physics_mode) {
                physics_mode->extend_layout(context.get_pdl_write());
                // Update config with physics mode requirements
                solver_config.set_use_pmass_field(physics_mode->uses_pmass_field());
            }
        }

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
        void clear_merged_pos_trees();
        void reset_presteps_rint();

        void reset_neighbors_cache();

        void init_ghost_layout();

        void reset_merge_ghosts_fields();

        void copy_density_to_patchdata();

        void update_sync_load_values();

        Solver(ShamrockCtx &context) : context(context) {}

        // ════════════════════════════════════════════════════════════════════════
        // Physics mode selection - each mode owns its config
        // Config structs are in physics/{newtonian,sr,mhd}/ subfolders
        // ════════════════════════════════════════════════════════════════════════

        /**
         * @brief Set Newtonian physics mode (default)
         * @param use_grad_h Enable grad-h correction (Price 2012)
         */
        void set_physics_newtonian(bool use_grad_h = false);

        /**
         * @brief Set Special Relativistic physics mode
         * @param c_speed Speed of light (default: 1.0 for natural units)
         */
        void set_physics_sr(Tscal c_speed = Tscal{1.0});

        /**
         * @brief Set MHD physics mode (placeholder)
         * @param resistivity Ohmic resistivity (0 = ideal MHD)
         */
        void set_physics_mhd(Tscal resistivity = Tscal{0.0});

        bool is_physics_newtonian() const;
        bool is_physics_sr() const;
        bool is_physics_mhd() const;

        /**
         * @brief Initialize physics mode if not already set (defaults to Newtonian)
         *
         * Called lazily on first evolve to allow config to be set before physics init.
         */
        void init_physics_mode();

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
