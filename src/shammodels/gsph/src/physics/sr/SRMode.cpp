// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SRMode.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothee David--Cleris (tim.shamrock@proton.me) --no git blame--
 * @brief Implementation of Special Relativistic physics mode for GSPH
 *
 * Owns the complete timestep sequence with SR-specific steps:
 *   predictor → boundary → tree → omega → gradients → eos
 *   → init_conserved → forces → recover_primitives → corrector
 *
 * Based on Kitajima et al. (2025) arXiv:2510.18251v1
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/modules/IterateSmoothingLengthVolume.hpp"
#include "shammodels/gsph/physics/sr/SREOS.hpp"
#include "shammodels/gsph/physics/sr/SRForceKernel.hpp"
#include "shammodels/gsph/physics/sr/SRMode.hpp"
#include "shammodels/gsph/physics/sr/SRPrimitiveRecovery.hpp"
#include "shammodels/gsph/physics/sr/SRTimestepper.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversalCache.hpp"

namespace shammodels::gsph::physics::sr {

    // ════════════════════════════════════════════════════════════════════════════
    // Core Interface - evolve_timestep owns the full sequence
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    typename SRMode<Tvec, SPHKernel>::Tscal SRMode<Tvec, SPHKernel>::evolve_timestep(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        Tscal dt,
        const core::SolverCallbacks<Tscal> &callbacks) {

        StackEntry stack_loc{};

        Tscal t_current = config.get_time();

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 1: PREDICTOR (S += dS*dt/2, e += de*dt/2, x += v*dt)
        // On first timestep, skip conserved update (dS=de=0), only do position drift
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
            // SR uses c_smooth in solver_config which the standard neighbor cache respects
            callbacks.start_neighbors();
            // SR-specific: omega/density with volume-based h
            h_converged = compute_omega_sr(storage, config, scheduler);

            if (!h_converged && hstep_cnt + 1 < hstep_max) {
                if (shamcomm::world_rank() == 0) {
                    shamcomm::logs::info_ln("SR", "h subcycle ", hstep_cnt + 1);
                }
                callbacks.reset_for_h_iteration();
            }
        }

        if (!h_converged) {
            shambase::throw_with_loc<std::runtime_error>(
                shambase::format("SR: h-iteration did not converge after {} subcycles", hstep_max));
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 4: PHYSICS SEQUENCE (SR-specific: includes primitive recovery)
        // ═══════════════════════════════════════════════════════════════════════
        callbacks.compute_gradients();
        callbacks.init_ghost_layout();
        callbacks.communicate_ghosts();
        compute_eos(storage, config, scheduler);
        callbacks.copy_density();

        // SR-GSPH: Initialize conserved variables from primitives on first timestep
        // Must happen AFTER EOS is computed (need P, ρ) but BEFORE forces
        bool first_timestep = !sr_initialized_;
        if (!sr_initialized_) {
            init_conserved(storage, config, scheduler);
        }

        // SR-specific: recover primitives before force computation
        recover_primitives(storage, config, scheduler);

        prepare_corrector(storage, config, scheduler);
        compute_forces(storage, config, scheduler);

        // Check for NaN in force derivatives - fail fast with helpful message
        check_derivatives_for_nan(storage, config, scheduler, "after compute_forces");

        // Update axyz for CFL timestep calculation
        // In SR, effective acceleration is dS/γH (since S = γHv)
        update_acceleration_for_cfl(storage, config, scheduler);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 5: CORRECTOR (S += dS*dt/2, e += de*dt/2)
        // Skip corrector on first real timestep (where dt was computed without
        // knowledge of actual forces). This allows CFL to properly constrain
        // the first actual integration step.
        // ═══════════════════════════════════════════════════════════════════════
        bool skip_corrector = first_timestep || (!first_real_step_done_ && dt > Tscal{0});
        if (!first_real_step_done_ && dt > Tscal{0}) {
            first_real_step_done_ = true;
        }

        if (!skip_corrector) {
            bool success = apply_corrector(storage, config, scheduler, dt);
            if (!success) {
                shamcomm::logs::warn_ln("SR", "Corrector detected superluminal particles");
            }
            // SR-specific: recover primitives after corrector
            recover_primitives(storage, config, scheduler);
        } else if (first_timestep) {
            // On very first timestep (dt=0), still recover primitives for consistency
            recover_primitives(storage, config, scheduler);
        }

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
    void SRMode<Tvec, SPHKernel>::init_fields(Storage &storage, Config &config) {
        StackEntry stack_loc{};

        // SR mode uses piecewise constant reconstruction (no gradients)
        config.set_use_gradients(false);

        using namespace shamrock::solvergraph;

        // c_speed already set from sr_config_ in constructor

        SolverGraph &solver_graph = storage.solver_graph;

        // SR-specific conserved variable fields
        if (!storage.S_momentum) {
            storage.S_momentum
                = solver_graph.register_edge("S_momentum", Field<Tvec>(1, "S_momentum", "S"));
        }

        if (!storage.e_energy) {
            storage.e_energy
                = solver_graph.register_edge("e_energy", Field<Tscal>(1, "e_energy", "e"));
        }

        if (!storage.dS_momentum) {
            storage.dS_momentum = solver_graph.register_edge(
                "dS_momentum", Field<Tvec>(1, "dS_momentum", "\\dot{S}"));
        }

        if (!storage.de_energy) {
            storage.de_energy
                = solver_graph.register_edge("de_energy", Field<Tscal>(1, "de_energy", "\\dot{e}"));
        }

        // Lorentz factor field for VTK output
        if (!storage.gamma_lorentz) {
            storage.gamma_lorentz = solver_graph.register_edge(
                "gamma_lorentz", Field<Tscal>(1, "gamma_lorentz", "\\gamma"));
        }

        // Register SR fields in field maps for physics-agnostic VTK output
        storage.scalar_fields["lorentz_factor"] = storage.gamma_lorentz;
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::init_conserved(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        SRPrimitiveRecovery<Tvec, SPHKernel>::init_conserved(storage, config, scheduler);
        sr_initialized_ = true;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Layout Extension - SR-specific fields
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::extend_layout(shamrock::patch::PatchDataLayerLayout &pdl) {
        // Per-particle mass for volume-based h (Kitajima et al. 2025)
        // Density computed as ρ = pmass × (hfact/h)³
        pdl.add_field<Tscal>("pmass", 1);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::extend_ghost_layout(
        shamrock::patch::PatchDataLayerLayout &ghost_layout) {
        // Per-particle mass needed for ghost density computation
        ghost_layout.add_field<Tscal>("pmass", 1);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // SR-specific Computations (volume-based h)
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    bool SRMode<Tvec, SPHKernel>::compute_omega_sr(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        StackEntry stack_loc{};

        // SR uses volume-based h iteration (IterateSmoothingLengthVolume)
        // Kitajima et al. (2025) Eq. 221, 230-233:
        //   V_p = 1 / Σ_j W(r_pj, h_p)  - Volume from kernel sum
        //   h_p = η × V_p^(1/d)         - h iterated from volume
        //   N = ν / V_p = ν × W_sum     - Density from baryon number
        //   C_smooth smooths h variation across discontinuities
        using namespace shamrock;
        using namespace shamrock::patch;
        using Kernel = SPHKernel<Tscal>;

        const Tscal pmass = config.gpart_mass;

        shamrock::solvergraph::Field<Tscal> &omega_field = shambase::get_check_ref(storage.omega);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);

        // Build sizes for field allocation (same pattern as ComputeOmega)
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes
            = std::make_shared<shamrock::solvergraph::Indexes<u32>>("sizes", "N");
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        omega_field.ensure_sizes(sizes->indexes);
        density_field.ensure_sizes(sizes->indexes);

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.template get_field_idx<Tscal>("hpart");

        auto &merged_xyzh = storage.merged_xyzh.get();

        // Position and h references from merged data
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> pos_merged
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("pos", "r");
        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_refs = {};

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hold
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_old", "h^{old}");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hold_refs = {};

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hnew
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_new", "h^{new}");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &mfield = merged_xyzh.get(p.id_patch);

            pos_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tvec>(0)));
            hold_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tscal>(1)));
            hnew_refs.add_obj(p.id_patch, std::ref(pdat.template get_field<Tscal>(ihpart)));
        });

        pos_merged->set_refs(pos_refs);
        hold->set_refs(hold_refs);
        hnew->set_refs(hnew_refs);

        // Copy h from merged to local (same as ComputeOmega)
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &mfield           = merged_xyzh.get(p.id_patch);
            auto &buf_hpart_merged = mfield.template get_field_buf_ref<Tscal>(1);
            auto &buf_hpart_local  = pdat.template get_field_buf_ref<Tscal>(ihpart);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_hpart_merged},
                sham::MultiRef{buf_hpart_local},
                cnt,
                [](u32 i, const Tscal *h_old, Tscal *h_new) {
                    h_new[i] = h_old[i];
                });
        });

        bool needs_cache_rebuild = false;

        // Volume-based h iteration (Kitajima Eq. 230-233)
        shamrock::SchedulerUtility utility(scheduler);
        ComputeField<Tscal> _epsilon_h = utility.make_compute_field<Tscal>("epsilon_h", 1);

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &eps_buf = _epsilon_h.get_buf_check(p.id_patch);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{eps_buf},
                cnt,
                [](u32 i, Tscal *eps) {
                    eps[i] = Tscal(1);
                });
        });

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> eps_h
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("eps_h", "\\epsilon_h");
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> eps_h_refs = {};
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = _epsilon_h.get_field(p.id_patch);
            eps_h_refs.add_obj(p.id_patch, std::ref(field));
        });
        eps_h->set_refs(eps_h_refs);

        // Volume-based h iteration for SR (Kitajima et al. 2025)
        auto vol_iter = std::make_shared<gsph::modules::IterateSmoothingLengthVolume<Tvec, Kernel>>(
            pmass, config.htol_up_coarse_cycle, config.htol_up_fine_cycle, config.c_smooth);
        vol_iter->set_edges(sizes, storage.neigh_cache, pos_merged, hold, hnew, eps_h);

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<bool>> is_converged
            = std::make_shared<shamrock::solvergraph::ScalarEdge<bool>>(
                "is_converged", "converged");

        shammodels::sph::modules::LoopSmoothingLengthIter<Tvec> loop_smth_h_iter(
            vol_iter, config.epsilon_h, config.h_iter_per_subcycles, false);
        loop_smth_h_iter.set_edges(eps_h, is_converged);

        loop_smth_h_iter.evaluate();

        if (!is_converged->value) {
            // Check for particles needing cache rebuild (eps < 0)
            u64 cnt_unconverged = 0;
            scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto res
                    = _epsilon_h.get_field(p.id_patch).get_ids_buf_where([](auto access, u32 id) {
                          return access[id] < Tscal(0);
                      });
                cnt_unconverged += std::get<1>(res);
            });
            u64 global_cnt_unconverged = shamalgs::collective::allreduce_sum(cnt_unconverged);

            if (global_cnt_unconverged > 0) {
                needs_cache_rebuild = true;
            }
        }

        // Compute density (Kitajima Eq. 221: N = ν × W_sum) and omega
        static constexpr Tscal Rkern = Kernel::Rkern;
        static constexpr u32 dim     = shambase::VectorProperties<Tvec>::dimension;
        auto &neigh_cache            = storage.neigh_cache->neigh_cache;

        const bool has_pmass = config.has_field_pmass();
        const u32 ipmass     = has_pmass ? pdl.template get_field_idx<Tscal>("pmass") : 0;

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &mfield = merged_xyzh.get(p.id_patch);
            auto &pcache = neigh_cache.get(p.id_patch);

            auto &buf_xyz   = mfield.template get_field_buf_ref<Tvec>(0);
            auto &buf_hpart = pdat.template get_field_buf_ref<Tscal>(ihpart);

            auto &dens_field = density_field.get_field(p.id_patch);
            auto &omeg_field = omega_field.get_field(p.id_patch);

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto ploop_ptrs  = pcache.get_read_access(depends_list);
            auto xyz_acc     = buf_xyz.get_read_access(depends_list);
            auto h_acc       = buf_hpart.get_read_access(depends_list);
            auto density_acc = dens_field.get_buf().get_write_access(depends_list);
            auto omega_acc   = omeg_field.get_buf().get_write_access(depends_list);

            const Tscal *pmass_acc                   = nullptr;
            sham::DeviceBuffer<Tscal> *buf_pmass_ptr = nullptr;
            if (has_pmass) {
                buf_pmass_ptr = &pdat.template get_field_buf_ref<Tscal>(ipmass);
                pmass_acc     = buf_pmass_ptr->get_read_access(depends_list);
            }

            // Kitajima volume-based density: N = ν × W_sum (Eq. 221)
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                shambase::parallel_for(cgh, cnt, "sr_compute_density_omega", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec xyz_a = xyz_acc[id_a];
                    Tscal h_a  = h_acc[id_a];
                    Tscal dint = h_a * h_a * Rkern * Rkern;

                    // Initialize with self-contribution (r=0)
                    // Kitajima Eq. 221: W_sum = Σ_j W(r_ij, h_i) includes self
                    Tscal W_self     = Kernel::W_3d(Tscal(0), h_a);
                    Tscal dW_dh_self = Kernel::dhW_3d(Tscal(0), h_a);
                    Tscal rho_sum    = has_pmass ? W_self : pmass * W_self;
                    Tscal sumdWdh    = has_pmass ? dW_dh_self : pmass * dW_dh_self;

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // Skip self (already counted above)
                        if (id_a == id_b)
                            return;

                        Tvec dr    = xyz_a - xyz_acc[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        if (has_pmass) {
                            // Kitajima: W_sum without mass weighting
                            rho_sum += Kernel::W_3d(rab, h_a);
                            sumdWdh += Kernel::dhW_3d(rab, h_a);
                        } else {
                            rho_sum += pmass * Kernel::W_3d(rab, h_a);
                            sumdWdh += pmass * Kernel::dhW_3d(rab, h_a);
                        }
                    });

                    if (has_pmass) {
                        // N = ν × W_sum (Kitajima Eq. 221)
                        Tscal nu_a = pmass_acc[id_a];
                        rho_sum *= nu_a;
                        sumdWdh *= nu_a;
                    }

                    density_acc[id_a] = sycl::max(rho_sum, Tscal(1e-30));

                    Tscal omega_val = Tscal(1);
                    if (rho_sum > Tscal(1e-30)) {
                        omega_val = Tscal(1) + h_a / (Tscal(dim) * rho_sum) * sumdWdh;
                        omega_val = sycl::clamp(omega_val, Tscal(0.5), Tscal(2.0));
                    }
                    omega_acc[id_a] = omega_val;
                });
            });

            pcache.complete_event_state({e});
            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            dens_field.get_buf().complete_event_state(e);
            omeg_field.get_buf().complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });

        return !needs_cache_rebuild;
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Internal Implementation - Time Integration
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::clear_derivatives(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        StackEntry stack_loc{};
        using namespace shamrock::patch;

        shamrock::solvergraph::Field<Tvec> &dS_field  = *storage.dS_momentum;
        shamrock::solvergraph::Field<Tscal> &de_field = *storage.de_energy;

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{buf_dS, buf_de},
                cnt,
                [](u32 i, Tvec *dS, Tscal *de) {
                    dS[i] = Tvec{0, 0, 0};
                    de[i] = Tscal{0};
                });
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::do_predictor(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {
        SRTimestepper<Tvec, SPHKernel>::do_predictor(storage, config, scheduler, dt);
    }

    template<class Tvec, template<class> class SPHKernel>
    bool SRMode<Tvec, SPHKernel>::apply_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {
        return SRTimestepper<Tvec, SPHKernel>::apply_corrector(storage, config, scheduler, dt);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::prepare_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        SRTimestepper<Tvec, SPHKernel>::prepare_corrector(storage, config, scheduler);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Internal Implementation - Physics Computations
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::compute_forces(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};

        riemann::ExactConfig exact_cfg;
        exact_cfg.tol      = config.sr_tol;
        exact_cfg.max_iter = config.sr_max_iter;

        SRForceKernel<Tvec, SPHKernel> force_kernel(scheduler, config, storage);
        force_kernel.compute_exact(exact_cfg);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::compute_eos(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        SREOS<Tvec, SPHKernel>::compute(storage, config, scheduler);
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::recover_primitives(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        SRPrimitiveRecovery<Tvec, SPHKernel>::recover(storage, config, scheduler);

        // Check for NaN in velocity field after recovery - fail fast with helpful error
        using namespace shamrock::patch;
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        bool has_nan = false;
        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            if (has_nan)
                return;

            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

            // Copy to host to check for NaN
            std::vector<Tvec> vxyz_host = buf_vxyz.copy_to_stdvec();

            for (u32 i = 0; i < cnt; ++i) {
                if (std::isnan(vxyz_host[i][0]) || std::isnan(vxyz_host[i][1])
                    || std::isnan(vxyz_host[i][2])) {
                    has_nan = true;

                    // Read conserved variables for diagnostics
                    shamrock::solvergraph::Field<Tvec> &S_field  = *storage.S_momentum;
                    shamrock::solvergraph::Field<Tscal> &e_field = *storage.e_energy;

                    sham::DeviceBuffer<Tvec> &buf_S  = S_field.get_buf(cur_p.id_patch);
                    sham::DeviceBuffer<Tscal> &buf_e = e_field.get_buf(cur_p.id_patch);

                    std::vector<Tvec> S_host  = buf_S.copy_to_stdvec();
                    std::vector<Tscal> e_host = buf_e.copy_to_stdvec();

                    Tscal S_mag = std::sqrt(
                        S_host[i][0] * S_host[i][0] + S_host[i][1] * S_host[i][1]
                        + S_host[i][2] * S_host[i][2]);

                    shamcomm::logs::err_ln(
                        "SR",
                        "NaN detected in primitive recovery at particle ",
                        i,
                        "\n  Conserved: S_mag=",
                        S_mag,
                        ", e=",
                        e_host[i],
                        "\n  This usually means the timestep is too large or the initial ",
                        "conditions produce unphysical conserved variables.",
                        "\n  Try reducing CFL or checking initial setup.");

                    break;
                }
            }
        });

        // Global reduction to check if any rank has NaN
        int local_nan  = has_nan ? 1 : 0;
        int global_nan = shamalgs::collective::allreduce_max(local_nan);

        if (global_nan > 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "SR-GSPH: NaN detected in primitive recovery. "
                "Check the log output above for details on the problematic particle.");
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::update_acceleration_for_cfl(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        StackEntry stack_loc{};

        // Update axyz field for CFL calculation
        // In SR, the momentum equation is: ν dS/dt = F (force per particle)
        // where S = γHv (conserved momentum per baryon).
        //
        // The acceleration is: dv/dt = dS/(γH) approximately
        // For CFL force constraint dt = C_force * sqrt(h / |a|), we need |dv/dt|.
        //
        // Using dS directly is WRONG because it gives:
        //   dt = C * sqrt(h/|dS|)  → Δv ∝ sqrt(|dS|) which is unbounded for large |dS|
        //
        // Instead, we use a = dS/(γH) which gives:
        //   dt = C * sqrt(h * γH / |dS|)  → Δv ∝ sqrt(|dS|/(γH)) which is bounded
        //
        // We compute γ from velocity and H from EOS.

        using namespace shamrock::patch;
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        shamrock::solvergraph::Field<Tvec> &dS_field = *storage.dS_momentum;
        shamrock::solvergraph::Field<Tscal> &pressure_field
            = shambase::get_check_ref(storage.pressure);
        shamrock::solvergraph::Field<Tscal> &density_field
            = shambase::get_check_ref(storage.density);

        const Tscal gamma_eos = config.get_eos_gamma();
        const Tscal c2        = config.c_speed * config.c_speed;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_dS   = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tvec> &buf_axyz = pdat.get_field_buf_ref<Tvec>(iaxyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tscal> &buf_P   = pressure_field.get_field(cur_p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &buf_rho = density_field.get_field(cur_p.id_patch).get_buf();

            sham::EventList depends_list;
            auto dS_acc   = buf_dS.get_read_access(depends_list);
            auto vxyz_acc = buf_vxyz.get_read_access(depends_list);
            auto P_acc    = buf_P.get_read_access(depends_list);
            auto rho_acc  = buf_rho.get_read_access(depends_list);
            auto axyz_acc = buf_axyz.get_write_access(depends_list);

            auto e = dev_sched->get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cnt, "SR-CFL-axyz", [=](u64 gid) {
                    u32 i = (u32) gid;

                    // Compute Lorentz factor from velocity
                    const Tvec v   = vxyz_acc[i];
                    const Tscal v2 = sycl::dot(v, v) / c2;
                    const Tscal gamma_lor
                        = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2, Tscal{1e-10}));

                    // Compute specific enthalpy H = 1 + u/c² + P/(ρc²)
                    // where u = P / ((γ-1)ρ)
                    const Tscal P   = sycl::fmax(P_acc[i], Tscal{1e-30});
                    const Tscal rho = sycl::fmax(rho_acc[i], Tscal{1e-30});
                    const Tscal u   = P / ((gamma_eos - Tscal{1}) * rho);
                    const Tscal H   = Tscal{1} + u / c2 + P / (rho * c2);

                    // Effective acceleration: a ≈ dS / (γH)
                    const Tscal gammaH = gamma_lor * H;
                    axyz_acc[i]        = dS_acc[i] / sycl::fmax(gammaH, Tscal{1});
                });
            });

            buf_dS.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_P.complete_event_state(e);
            buf_rho.complete_event_state(e);
            buf_axyz.complete_event_state(e);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRMode<Tvec, SPHKernel>::check_derivatives_for_nan(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const std::string &context) {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");
        const bool has_pmass      = config.has_field_pmass();
        const u32 ipmass          = has_pmass ? pdl.get_field_idx<Tscal>("pmass") : 0;

        shamrock::solvergraph::Field<Tvec> &dS_field  = *storage.dS_momentum;
        shamrock::solvergraph::Field<Tscal> &de_field = *storage.de_energy;
        shamrock::solvergraph::Field<Tscal> &P_field  = shambase::get_check_ref(storage.pressure);

        bool has_nan = false;
        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            if (has_nan)
                return;

            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_dS   = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de  = de_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tscal> &buf_h   = pdat.get_field_buf_ref<Tscal>(ihpart);
            sham::DeviceBuffer<Tscal> &buf_P   = P_field.get_buf(cur_p.id_patch);

            std::vector<Tvec> dS_host   = buf_dS.copy_to_stdvec();
            std::vector<Tscal> de_host  = buf_de.copy_to_stdvec();
            std::vector<Tvec> xyz_host  = buf_xyz.copy_to_stdvec();
            std::vector<Tvec> vxyz_host = buf_vxyz.copy_to_stdvec();
            std::vector<Tscal> h_host   = buf_h.copy_to_stdvec();
            std::vector<Tscal> P_host   = buf_P.copy_to_stdvec();

            std::vector<Tscal> pmass_host;
            if (has_pmass) {
                sham::DeviceBuffer<Tscal> &buf_pmass = pdat.get_field_buf_ref<Tscal>(ipmass);
                pmass_host                           = buf_pmass.copy_to_stdvec();
            }

            for (u32 i = 0; i < cnt; ++i) {
                if (std::isnan(dS_host[i][0]) || std::isnan(dS_host[i][1])
                    || std::isnan(dS_host[i][2]) || std::isnan(de_host[i])
                    || std::isinf(dS_host[i][0]) || std::isinf(dS_host[i][1])
                    || std::isinf(dS_host[i][2]) || std::isinf(de_host[i])) {
                    has_nan = true;

                    Tscal dS_mag = std::sqrt(
                        dS_host[i][0] * dS_host[i][0] + dS_host[i][1] * dS_host[i][1]
                        + dS_host[i][2] * dS_host[i][2]);
                    Tscal v_mag = std::sqrt(
                        vxyz_host[i][0] * vxyz_host[i][0] + vxyz_host[i][1] * vxyz_host[i][1]
                        + vxyz_host[i][2] * vxyz_host[i][2]);
                    Tscal pmass_val = has_pmass ? pmass_host[i] : config.gpart_mass;

                    shamcomm::logs::err_ln(
                        "SR",
                        "NaN/Inf detected in derivatives ",
                        context,
                        " at particle ",
                        i,
                        "\n  dS = (",
                        dS_host[i][0],
                        ", ",
                        dS_host[i][1],
                        ", ",
                        dS_host[i][2],
                        ") |dS|=",
                        dS_mag,
                        "\n  de = ",
                        de_host[i],
                        "\n  xyz = (",
                        xyz_host[i][0],
                        ", ",
                        xyz_host[i][1],
                        ", ",
                        xyz_host[i][2],
                        ")",
                        "\n  |v| = ",
                        v_mag,
                        ", h = ",
                        h_host[i],
                        ", pmass = ",
                        pmass_val,
                        ", P = ",
                        P_host[i],
                        "\n  Likely causes: Riemann solver divergence, extreme pressure ratio, "
                        "or particles too close.");

                    break;
                }
            }
        });

        if (has_nan) {
            shambase::throw_with_loc<std::runtime_error>(
                "SR-GSPH: NaN/Inf detected in force derivatives. Check the log output above for "
                "details.");
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Explicit Template Instantiations
    // ════════════════════════════════════════════════════════════════════════════

    using namespace shammath;

    template class SRMode<sycl::vec<double, 3>, M4>;
    template class SRMode<sycl::vec<double, 3>, M6>;
    template class SRMode<sycl::vec<double, 3>, M8>;
    template class SRMode<sycl::vec<double, 3>, C2>;
    template class SRMode<sycl::vec<double, 3>, C4>;
    template class SRMode<sycl::vec<double, 3>, C6>;
    template class SRMode<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::sr
