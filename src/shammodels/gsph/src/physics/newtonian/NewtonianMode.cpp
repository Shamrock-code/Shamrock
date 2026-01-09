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
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianEOS.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianFieldNames.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianForceKernel.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianMode.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianTimestepper.hpp"
#include "shammodels/gsph/physics/newtonian/riemann/RiemannBase.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/modules/LoopSmoothingLengthIter.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversalCache.hpp"

namespace shammodels::gsph::physics::newtonian {

    // ════════════════════════════════════════════════════════════════════════════
    // Core Interface - evolve_timestep owns the full sequence
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    typename NewtonianMode<Tvec, SPHKernel>::Tscal NewtonianMode<Tvec, SPHKernel>::evolve_timestep(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {
        using namespace shamrock::solvergraph;

        StackEntry stack_loc{};

        // Access solvergraph for node evaluation
        auto &graph = storage.solver_graph;

        // Helper to evaluate a node by name
        auto eval = [&graph](const char *name) {
            graph.get_node_ptr_base(name)->evaluate();
        };

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 1: PREDICTOR (v += a*dt/2, x += v*dt)
        // ═══════════════════════════════════════════════════════════════════════
        do_predictor(storage, config, scheduler, dt);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 2: BOUNDARY CONDITIONS
        // ═══════════════════════════════════════════════════════════════════════
        eval("gen_serial_patch_tree");
        eval("apply_boundary");

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 3: H-ITERATION LOOP (tree build + density/omega)
        // Newtonian uses its own h-iteration (no c_smooth, standard mass-based SPH)
        // ═══════════════════════════════════════════════════════════════════════
        u32 hstep_cnt    = 0;
        u32 hstep_max    = config.h_max_subcycles_count;
        bool h_converged = false;

        for (; hstep_cnt < hstep_max && !h_converged; hstep_cnt++) {
            eval("gen_ghost_handler");
            eval("build_ghost_cache");
            eval("merge_position_ghost");
            eval("build_trees");
            eval("compute_presteps");
            eval("start_neighbors");

            // Use Newtonian-specific h-iteration (no c_smooth, standard mass-based SPH)
            h_converged = compute_omega_newtonian(storage, config, scheduler);

            if (!h_converged && hstep_cnt + 1 < hstep_max) {
                if (shamcomm::world_rank() == 0) {
                    shamcomm::logs::info_ln("Newtonian", "h subcycle ", hstep_cnt + 1);
                }
                eval("reset_for_h_iteration");
            }
        }

        if (!h_converged) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "Newtonian: h-iteration did not converge after {} subcycles", hstep_max));
        }

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 4: PHYSICS SEQUENCE
        // ═══════════════════════════════════════════════════════════════════════
        eval("compute_gradients");
        eval("init_ghost_layout");
        eval("communicate_ghosts");
        compute_eos(storage, config, scheduler);
        eval("copy_density");
        prepare_corrector(storage, config, scheduler);
        compute_forces(storage, config, scheduler);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 5: CORRECTOR (v += a*dt/2)
        // ═══════════════════════════════════════════════════════════════════════
        apply_corrector(storage, config, scheduler, dt);

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 6: CFL TIMESTEP
        // ═══════════════════════════════════════════════════════════════════════
        eval("compute_dt");
        Tscal dt_next = graph.template get_edge_ref<IDataEdge<Tscal>>("dt_next").data;

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 7: CLEANUP
        // ═══════════════════════════════════════════════════════════════════════
        eval("cleanup");

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

        // Dispatch based on Riemann solver config
        using RiemannCfg        = typename Config::RiemannConfig;
        const auto &riemann_cfg = config.riemann_config;

        if (const auto *v = std::get_if<typename RiemannCfg::Iterative>(&riemann_cfg.config)) {
            riemann::IterativeConfig cfg;
            cfg.tol      = v->tol;
            cfg.max_iter = v->max_iter;
            compute_forces_iterative(storage, config, scheduler, cfg);
        } else if (std::get_if<typename RiemannCfg::HLL>(&riemann_cfg.config)) {
            riemann::HLLConfig cfg;
            compute_forces_hll(storage, config, scheduler, cfg);
        } else if (std::get_if<typename RiemannCfg::HLLC>(&riemann_cfg.config)) {
            riemann::HLLCConfig cfg;
            compute_forces_hllc(storage, config, scheduler, cfg);
        } else {
            shambase::throw_unimplemented("Unsupported Riemann solver type for Newtonian mode");
        }
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

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianMode<Tvec, SPHKernel>::compute_forces_hllc(
        Storage &storage,
        const Config &config,
        PatchScheduler &scheduler,
        const riemann::HLLCConfig &riemann_config) {
        NewtonianForceKernel<Tvec, SPHKernel> force_kernel(scheduler, config, storage);
        force_kernel.compute_hllc(riemann_config);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Newtonian-specific h-iteration and density computation
    // ════════════════════════════════════════════════════════════════════════════

    template<class Tvec, template<class> class SPHKernel>
    bool NewtonianMode<Tvec, SPHKernel>::compute_omega_newtonian(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        using Kernel = SPHKernel<Tscal>;

        const Tscal pmass = config.gpart_mass;

        solvergraph::Field<Tscal> &omega_field   = shambase::get_check_ref(storage.omega);
        solvergraph::Field<Tscal> &density_field = shambase::get_check_ref(storage.density);

        // Build sizes for field allocation
        std::shared_ptr<solvergraph::Indexes<u32>> sizes
            = std::make_shared<solvergraph::Indexes<u32>>("sizes", "N");
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        omega_field.ensure_sizes(sizes->indexes);
        density_field.ensure_sizes(sizes->indexes);

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ihpart          = pdl.template get_field_idx<Tscal>("hpart");

        auto &merged_xyzh = storage.merged_xyzh.get();

        // Position and h references from merged data
        std::shared_ptr<solvergraph::FieldRefs<Tvec>> pos_merged
            = std::make_shared<solvergraph::FieldRefs<Tvec>>("pos", "r");
        solvergraph::DDPatchDataFieldRef<Tvec> pos_refs = {};

        std::shared_ptr<solvergraph::FieldRefs<Tscal>> hold
            = std::make_shared<solvergraph::FieldRefs<Tscal>>("h_old", "h^{old}");
        solvergraph::DDPatchDataFieldRef<Tscal> hold_refs = {};

        std::shared_ptr<solvergraph::FieldRefs<Tscal>> hnew
            = std::make_shared<solvergraph::FieldRefs<Tscal>>("h_new", "h^{new}");
        solvergraph::DDPatchDataFieldRef<Tscal> hnew_refs = {};

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &mfield = merged_xyzh.get(p.id_patch);

            pos_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tvec>(0)));
            hold_refs.add_obj(p.id_patch, std::ref(mfield.template get_field<Tscal>(1)));
            hnew_refs.add_obj(p.id_patch, std::ref(pdat.template get_field<Tscal>(ihpart)));
        });

        pos_merged->set_refs(pos_refs);
        hold->set_refs(hold_refs);
        hnew->set_refs(hnew_refs);

        // Copy h from merged to local
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

        // Epsilon tracking for convergence
        SchedulerUtility utility(scheduler);
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

        std::shared_ptr<solvergraph::FieldRefs<Tscal>> eps_h
            = std::make_shared<solvergraph::FieldRefs<Tscal>>("eps_h", "\\epsilon_h");
        solvergraph::DDPatchDataFieldRef<Tscal> eps_h_refs = {};
        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &field = _epsilon_h.get_field(p.id_patch);
            eps_h_refs.add_obj(p.id_patch, std::ref(field));
        });
        eps_h->set_refs(eps_h_refs);

        // Standard density-based h iteration (NO c_smooth for Newtonian!)
        // Uses default c_smooth=1.0 from IterateSmoothingLengthDensity constructor
        auto std_iter = std::make_shared<sph::modules::IterateSmoothingLengthDensity<Tvec, Kernel>>(
            config.gpart_mass, config.htol_up_coarse_cycle, config.htol_up_fine_cycle);
        std_iter->set_edges(sizes, storage.neigh_cache, pos_merged, hold, hnew, eps_h);

        std::shared_ptr<solvergraph::ScalarEdge<bool>> is_converged
            = std::make_shared<solvergraph::ScalarEdge<bool>>("is_converged", "converged");

        sph::modules::LoopSmoothingLengthIter<Tvec> loop_smth_h_iter(
            std_iter, config.epsilon_h, config.h_iter_per_subcycles, false);
        loop_smth_h_iter.set_edges(eps_h, is_converged);

        loop_smth_h_iter.evaluate();

        bool needs_cache_rebuild = false;
        if (!is_converged->value) {
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

        // Compute density and omega
        static constexpr Tscal Rkern = Kernel::Rkern;
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

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                shambase::parallel_for(cgh, cnt, "newtonian_compute_density_omega", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tvec xyz_a = xyz_acc[id_a];
                    Tscal h_a  = h_acc[id_a];
                    Tscal dint = h_a * h_a * Rkern * Rkern;

                    Tscal rho_sum = Tscal(0);
                    Tscal sumdWdh = Tscal(0);

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        Tvec dr    = xyz_a - xyz_acc[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        // Standard mass-based SPH density (NO c_smooth)
                        Tscal m_b = has_pmass ? pmass_acc[id_b] : pmass;
                        rho_sum += m_b * Kernel::W_3d(rab, h_a);
                        sumdWdh += m_b * Kernel::dhW_3d(rab, h_a);
                    });

                    // Clamp density to avoid numerical issues
                    rho_sum = sycl::fmax(rho_sum, Tscal{1e-30});

                    // Omega = 1 + (h/3rho) * d(rho)/dh
                    Tscal omega_val = Tscal(1) + (h_a / (Tscal(3) * rho_sum)) * sumdWdh;

                    density_acc[id_a] = rho_sum;
                    omega_acc[id_a]   = omega_val;
                });
            });

            pcache.complete_event_state(e);
            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            dens_field.get_buf().complete_event_state(e);
            omeg_field.get_buf().complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }
        });

        return is_converged->value && !needs_cache_rebuild;
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
