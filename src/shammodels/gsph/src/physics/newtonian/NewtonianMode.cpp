// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NewtonianMode.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Implementation of Newtonian physics mode for GSPH
 *
 * Owns the complete timestep sequence:
 *   predictor → boundary → tree → omega → gradients → eos → forces → corrector
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianEOS.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianFieldNames.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianForceKernel.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianMode.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianTimestepper.hpp"
#include "shammodels/gsph/physics/newtonian/riemann/RiemannBase.hpp"

namespace shammodels::gsph::physics::newtonian {

    // ════════════════════════════════════════════════════════════════════════════
    // Core Interface - evolve_timestep owns the full sequence
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    typename NewtonianMode<Tvec, SPHKernel>::Tscal NewtonianMode<Tvec, SPHKernel>::evolve_timestep(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        Tscal dt,
        const core::SolverCallbacks<Tscal> &callbacks) {

        StackEntry stack_loc{};

        Tscal t_current = config.get_time();

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 1: PREDICTOR (v += a*dt/2, x += v*dt)
        // ═══════════════════════════════════════════════════════════════════════
        do_predictor(storage, config, scheduler, dt);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 2: BOUNDARY CONDITIONS
        // ═══════════════════════════════════════════════════════════════════════
        callbacks.gen_serial_patch_tree();

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 3: H-ITERATION LOOP (tree build + density/omega)
        // ═══════════════════════════════════════════════════════════════════════
        u32 hstep_cnt    = 0;
        u32 hstep_max    = callbacks.h_max_subcycles;
        bool h_converged = false;

        for (; hstep_cnt < hstep_max && !h_converged; hstep_cnt++) {
            callbacks.gen_ghost_handler(t_current + dt);
            callbacks.build_ghost_cache();
            callbacks.merge_position_ghost();
            callbacks.build_trees();
            callbacks.compute_presteps();
            callbacks.start_neighbors();
            h_converged = callbacks.compute_omega();

            if (!h_converged && hstep_cnt + 1 < hstep_max) {
                if (shamcomm::world_rank() == 0) {
                    shamcomm::logs::info_ln("Newtonian", "h subcycle ", hstep_cnt + 1);
                }
                callbacks.reset_for_h_iteration();
            }
        }

        if (!h_converged) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "Newtonian: h-iteration did not converge after {} subcycles", hstep_max));
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 4: PHYSICS SEQUENCE
        // ═══════════════════════════════════════════════════════════════════════
        callbacks.compute_gradients();
        callbacks.init_ghost_layout();
        callbacks.communicate_ghosts();
        compute_eos(storage, config, scheduler);
        callbacks.copy_density();
        prepare_corrector(storage, config, scheduler);
        compute_forces(storage, config, scheduler);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 5: CORRECTOR (v += a*dt/2)
        // ═══════════════════════════════════════════════════════════════════════
        apply_corrector(storage, config, scheduler, dt);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 6: CFL TIMESTEP
        // ═══════════════════════════════════════════════════════════════════════
        Tscal dt_next = callbacks.compute_cfl();

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 7: CLEANUP
        // ═══════════════════════════════════════════════════════════════════════
        callbacks.cleanup();

        return dt_next;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::init_fields(Storage &storage, Config &config) {
        using namespace shamrock::solvergraph;
        SolverGraph &solver_graph = storage.solver_graph;

        // Newtonian mode currently uses piecewise constant (no gradient reconstruction)
        // so disable gradient communication in ghost layout
        config.set_use_gradients(false);

        // Register Newtonian density field
        if (!storage.density) {
            storage.density = solver_graph.register_edge(
                fields::DENSITY, Field<Tscal>(1, fields::DENSITY, "\\rho"));
        }

        // Register in scalar_fields for VTK output
        storage.scalar_fields[fields::DENSITY] = storage.density;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Internal Implementation - Time Integration
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::do_predictor(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {
        NewtonianTimestepper<Tvec, SPHKernel>::do_predictor(storage, config, scheduler, dt);
    }

    template<class Tvec, template<class> class SPHKernel>
    bool NewtonianMode<Tvec, SPHKernel>::apply_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {
        return NewtonianTimestepper<Tvec, SPHKernel>::apply_corrector(
            storage, config, scheduler, dt);
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::prepare_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        NewtonianTimestepper<Tvec, SPHKernel>::prepare_corrector(storage, config, scheduler);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Internal Implementation - Physics Computations
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::compute_forces(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};

        // Use default iterative Riemann solver (Phase 4: config will be owned by mode)
        riemann::IterativeConfig cfg;
        cfg.tol      = Tscal{1e-6};
        cfg.max_iter = 20;
        compute_forces_iterative(storage, config, scheduler, cfg);
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::compute_eos(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        NewtonianEOS<Tvec, SPHKernel>::compute(storage, config, scheduler);
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::compute_forces_iterative(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const riemann::IterativeConfig &riemann_config) {
        NewtonianForceKernel<Tvec, SPHKernel> force_kernel(scheduler, config, storage);
        force_kernel.compute_iterative(riemann_config);
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::compute_forces_hll(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const riemann::HLLConfig &riemann_config) {
        NewtonianForceKernel<Tvec, SPHKernel> force_kernel(scheduler, config, storage);
        force_kernel.compute_hll(riemann_config);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Explicit Template Instantiations
    // ════════════════════════════════════════════════════════════════════════════

    using namespace shammath;

    template class NewtonianMode<sycl::vec<double, 3>, M4>;
    template class NewtonianMode<sycl::vec<double, 3>, M6>;
    template class NewtonianMode<sycl::vec<double, 3>, M8>;
    template class NewtonianMode<sycl::vec<double, 3>, C2>;
    template class NewtonianMode<sycl::vec<double, 3>, C4>;
    template class NewtonianMode<sycl::vec<double, 3>, C6>;
    template class NewtonianMode<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::newtonian
