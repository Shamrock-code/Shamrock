// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file UpdateDerivs.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Implementation of GSPH derivative update module
 *
 * This file implements the core GSPH algorithm: for each particle pair,
 * we solve a 1D Riemann problem and use the result to compute forces.
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 *
 * This implementation follows:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shammodels/gsph/modules/UpdateDerivs.hpp"
#include "shambackends/math.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/math/forces.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"
#include "shammodels/gsph/math/sr/exact.hpp"
#include "shammodels/gsph/math/sr/forces.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversal.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs() {
    StackEntry stack_loc{};

    Cfg_Riemann cfg_riemann = solver_config.riemann_config;

    // Check if SR mode is enabled - use SR-specific update with exact Riemann solver
    if (solver_config.is_sr_enabled()) {
        update_derivs_sr();
        return;
    }

    // Standard Newtonian GSPH
    if (Iterative *v = std::get_if<Iterative>(&cfg_riemann.config)) {
        update_derivs_iterative(*v);
    } else if (HLLC *v = std::get_if<HLLC>(&cfg_riemann.config)) {
        update_derivs_hllc(*v);
    } else {
        shambase::throw_unimplemented("Riemann solver type not supported by UpdateDerivs");
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_iterative(
    Iterative cfg) {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    // Get field indices from the patch data layout
    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    // Optional internal energy fields (for adiabatic EOS)
    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
    const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

    // Ghost layout for neighbor data
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>("omega");
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
    u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    // Get merged data and caches from storage
    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega_field  = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    // Get pressure and soundspeed from storage (includes ghosts)
    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);
    shamrock::solvergraph::Field<Tscal> &soundspeed_field
        = shambase::get_check_ref(storage.soundspeed);

    // Iterate over all non-empty patches
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        // Get buffers for local and ghost data
        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = pressure_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = soundspeed_field.get_field(cur_p.id_patch).get_buf();

        // Get neighbor cache for this patch
        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        // Set up SYCL queue and event tracking
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        // Get density from merged ghost data (SPH summation density)
        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

        // Get buffer accessors
        auto xyz          = buf_xyz.get_read_access(depends_list);
        auto axyz         = buf_axyz.get_write_access(depends_list);
        auto vxyz         = buf_vxyz.get_read_access(depends_list);
        auto hpart        = buf_hpart.get_read_access(depends_list);
        auto omega_acc    = buf_omega.get_read_access(depends_list);
        auto density_acc  = buf_density.get_read_access(depends_list);
        auto pressure_acc = buf_pressure.get_read_access(depends_list);
        auto cs_acc       = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs   = pcache.get_read_access(depends_list);

        // Optional: internal energy
        sham::DeviceBuffer<Tscal> *buf_duint_ptr = nullptr;
        Tscal *duint_acc                         = nullptr;
        if (has_uint) {
            buf_duint_ptr = &pdat.get_field_buf_ref<Tscal>(iduint);
            duint_acc     = buf_duint_ptr->get_write_access(depends_list);
        }

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal gamma    = solver_config.get_eos_gamma();
            const Tscal tol      = cfg.tol;
            const u32 max_iter   = cfg.max_iter;
            const bool do_energy = has_uint;

            // Use shamrock's ObjectCacheIterator for neighbor traversal
            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "GSPH derivs iterative", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                // Initialize accumulators
                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                // Particle a state
                const Tscal h_a     = hpart[id_a];
                const Tvec xyz_a    = xyz[id_a];
                const Tvec vxyz_a   = vxyz[id_a];
                const Tscal omega_a = omega_acc[id_a];

                // Use SPH-summation density
                const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));

                // Use pressure and soundspeed from storage
                const Tscal P_a  = sycl::max(pressure_acc[id_a], Tscal(1e-30));
                const Tscal cs_a = sycl::max(cs_acc[id_a], Tscal(1e-10));

                // Loop over neighbors
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    if (id_a == id_b)
                        return; // Skip self

                    // Distance and kernel support check
                    const Tvec dr    = xyz_a - xyz[id_b];
                    const Tscal rab2 = sycl::dot(dr, dr);
                    const Tscal h_b  = hpart[id_b];

                    // Skip if outside kernel support
                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    const Tscal rab     = sycl::sqrt(rab2);
                    const Tvec vxyz_b   = vxyz[id_b];
                    const Tscal omega_b = omega_acc[id_b];

                    // Use SPH-summation density
                    const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));

                    // Use pressure and soundspeed from storage
                    const Tscal P_b  = sycl::max(pressure_acc[id_b], Tscal(1e-30));
                    const Tscal cs_b = sycl::max(cs_acc[id_b], Tscal(1e-10));

                    // Unit vector from a to b
                    const Tscal rab_inv  = sham::inv_sat_positive(rab);
                    const Tvec r_ab_unit = dr * rab_inv;

                    // Project velocities onto pair axis for 1D Riemann problem
                    const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                    const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                    // Solve 1D Riemann problem using iterative solver
                    // Convention: Left state = neighbor b, Right state = current a
                    auto riemann_result = riemann::iterative_solver<Tscal>(
                        u_b_proj,
                        rho_b,
                        P_b, // Left = neighbor
                        u_a_proj,
                        rho_a,
                        P_a, // Right = current
                        gamma,
                        tol,
                        max_iter);
                    const Tscal p_star = riemann_result.p_star;
                    const Tscal v_star = riemann_result.v_star;

                    // Kernel gradients
                    const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // GSPH force contribution
                    shammodels::gsph::add_gsph_force_contribution<Tvec, Tscal>(
                        pmass,
                        p_star,
                        v_star,
                        rho_a,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        r_ab_unit,
                        vxyz_a,
                        sum_axyz,
                        sum_du_a);
                });

                // Write accumulated derivatives
                axyz[id_a] = sum_axyz;
                if (duint_acc != nullptr) {
                    duint_acc[id_a] = sum_du_a;
                }
            });
        });

        // Complete event states for all buffers
        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_density.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);

        if (has_uint && buf_duint_ptr) {
            buf_duint_ptr->complete_event_state(e);
        }

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_hllc(HLLC cfg) {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    // Get field indices
    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    const bool has_uint = solver_config.has_field_uint();
    const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
    const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

    // Ghost layout
    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf   = ghost_layout.get_field_idx<Tscal>("omega");
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");
    u32 iuint_interf    = has_uint ? ghost_layout.get_field_idx<Tscal>("uint") : 0;

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega_field  = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    // Get pressure and soundspeed from storage (includes ghosts)
    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);
    shamrock::solvergraph::Field<Tscal> &soundspeed_field
        = shambase::get_check_ref(storage.soundspeed);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = pressure_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = soundspeed_field.get_field(cur_p.id_patch).get_buf();

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        // Get density from merged ghost data
        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

        auto xyz          = buf_xyz.get_read_access(depends_list);
        auto axyz         = buf_axyz.get_write_access(depends_list);
        auto vxyz         = buf_vxyz.get_read_access(depends_list);
        auto hpart        = buf_hpart.get_read_access(depends_list);
        auto omega_acc    = buf_omega.get_read_access(depends_list);
        auto density_acc  = buf_density.get_read_access(depends_list);
        auto pressure_acc = buf_pressure.get_read_access(depends_list);
        auto cs_acc       = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs   = pcache.get_read_access(depends_list);

        sham::DeviceBuffer<Tscal> *buf_duint_ptr = nullptr;
        Tscal *duint_acc                         = nullptr;
        if (has_uint) {
            buf_duint_ptr = &pdat.get_field_buf_ref<Tscal>(iduint);
            duint_acc     = buf_duint_ptr->get_write_access(depends_list);
        }

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal gamma    = solver_config.get_eos_gamma();
            const bool do_energy = has_uint;

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "GSPH derivs HLLC", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                const Tscal h_a     = hpart[id_a];
                const Tvec xyz_a    = xyz[id_a];
                const Tvec vxyz_a   = vxyz[id_a];
                const Tscal omega_a = omega_acc[id_a];

                // Use SPH-summation density
                const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));

                // Use pressure and soundspeed from storage
                const Tscal P_a  = sycl::max(pressure_acc[id_a], Tscal(1e-30));
                const Tscal cs_a = sycl::max(cs_acc[id_a], Tscal(1e-10));

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    if (id_a == id_b)
                        return;

                    const Tvec dr    = xyz_a - xyz[id_b];
                    const Tscal rab2 = sycl::dot(dr, dr);
                    const Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    const Tscal rab     = sycl::sqrt(rab2);
                    const Tvec vxyz_b   = vxyz[id_b];
                    const Tscal omega_b = omega_acc[id_b];

                    // Use SPH-summation density
                    const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));

                    // Use pressure and soundspeed from storage
                    const Tscal P_b  = sycl::max(pressure_acc[id_b], Tscal(1e-30));
                    const Tscal cs_b = sycl::max(cs_acc[id_b], Tscal(1e-10));

                    const Tscal rab_inv  = sham::inv_sat_positive(rab);
                    const Tvec r_ab_unit = dr * rab_inv;

                    // Project velocities onto pair axis for 1D Riemann problem
                    const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                    const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                    // Use HLLC approximate Riemann solver
                    // Convention: Left state = neighbor b, Right state = current a
                    auto riemann_result = riemann::hllc_solver<Tscal>(
                        u_b_proj,
                        rho_b,
                        P_b, // Left = neighbor
                        u_a_proj,
                        rho_a,
                        P_a, // Right = current
                        gamma);
                    const Tscal p_star = riemann_result.p_star;
                    const Tscal v_star = riemann_result.v_star;

                    const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // GSPH force contribution
                    shammodels::gsph::add_gsph_force_contribution<Tvec, Tscal>(
                        pmass,
                        p_star,
                        v_star,
                        rho_a,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        r_ab_unit,
                        vxyz_a,
                        sum_axyz,
                        sum_du_a);
                });

                axyz[id_a] = sum_axyz;
                if (duint_acc != nullptr) {
                    duint_acc[id_a] = sum_du_a;
                }
            });
        });

        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_density.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);

        if (has_uint && buf_duint_ptr) {
            buf_duint_ptr->complete_event_state(e);
        }

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_sr() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf   = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 ivxyz_interf    = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 idensity_interf = ghost_layout.get_field_idx<Tscal>("density");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    shamrock::solvergraph::Field<Tscal> &pressure_field = shambase::get_check_ref(storage.pressure);
    shamrock::solvergraph::Field<Tscal> &soundspeed_field
        = shambase::get_check_ref(storage.soundspeed);

    // Get SR conserved variable derivative fields from storage
    shamrock::solvergraph::Field<Tvec> &dS_field  = shambase::get_check_ref(storage.dS_momentum);
    shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage.de_energy);

    // SR config parameters
    const Tscal c_speed   = solver_config.sr_config.get_c_speed();
    const Tscal gamma_eos = solver_config.get_eos_gamma();
    const Tscal pmass     = solver_config.gpart_mass;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = pressure_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = soundspeed_field.get_field(cur_p.id_patch).get_buf();

        // Get SR derivative buffers from storage fields
        sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_de = de_field.get_field(cur_p.id_patch).get_buf();

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

        auto xyz          = buf_xyz.get_read_access(depends_list);
        auto vxyz         = buf_vxyz.get_read_access(depends_list);
        auto hpart        = buf_hpart.get_read_access(depends_list);
        auto density_acc  = buf_density.get_read_access(depends_list);
        auto pressure_acc = buf_pressure.get_read_access(depends_list);
        auto cs_acc       = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs   = pcache.get_read_access(depends_list);

        // Write access for SR derivative fields
        auto dS_acc = buf_dS.get_write_access(depends_list);
        auto de_acc = buf_de.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal c2 = c_speed * c_speed;

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "SR-GSPH derivs exact", [=](u64 gid) {
                u32 id_a = (u32) gid;

                Tvec sum_dS  = {0, 0, 0}; // dS/dt accumulator (momentum)
                Tscal sum_de = 0;         // de/dt accumulator (energy)

                const Tscal h_a   = hpart[id_a];
                const Tvec xyz_a  = xyz[id_a];
                const Tvec vxyz_a = vxyz[id_a];

                // SPH density summation gives lab-frame N = ν·ΣW (NOT rest-frame n)
                const Tscal N_a = sycl::fmax(density_acc[id_a], Tscal{1e-30});
                const Tscal P_a = sycl::fmax(pressure_acc[id_a], Tscal{1e-30});

                // Compute Lorentz factor
                const Tscal v2_a = sycl::dot(vxyz_a, vxyz_a) / c2;
                const Tscal gamma_a
                    = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2_a, Tscal{1e-10}));

                // Convert lab-frame N to rest-frame n = N/γ for thermodynamics
                const Tscal n_a = N_a / gamma_a;

                // Specific enthalpy: H = 1 + u/c² + P/(nc²) (uses rest-frame n)
                const Tscal u_a = P_a / ((gamma_eos - Tscal{1}) * n_a);
                const Tscal H_a = Tscal{1} + u_a / c2 + P_a / (n_a * c2);

                // Volume: V = m/N (uses lab-frame N)
                const Tscal V_a = pmass / sycl::fmax(N_a, Tscal{1e-30});

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    if (id_a == id_b)
                        return;

                    const Tvec dr    = xyz_a - xyz[id_b];
                    const Tscal rab2 = sycl::dot(dr, dr);
                    const Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    const Tscal rab   = sycl::sqrt(rab2);
                    const Tvec vxyz_b = vxyz[id_b];

                    // SPH density gives lab-frame N for particle b
                    const Tscal N_b = sycl::fmax(density_acc[id_b], Tscal{1e-30});
                    const Tscal P_b = sycl::fmax(pressure_acc[id_b], Tscal{1e-30});

                    // Relativistic quantities for particle b
                    const Tscal v2_b = sycl::dot(vxyz_b, vxyz_b) / c2;
                    const Tscal gamma_b
                        = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2_b, Tscal{1e-10}));

                    // Convert lab-frame N to rest-frame n = N/γ
                    const Tscal n_b = N_b / gamma_b;
                    const Tscal u_b = P_b / ((gamma_eos - Tscal{1}) * n_b);
                    const Tscal H_b = Tscal{1} + u_b / c2 + P_b / (n_b * c2);
                    const Tscal V_b = pmass / sycl::fmax(N_b, Tscal{1e-30});

                    // Unit vector from a to b
                    const Tscal rab_inv = sham::inv_sat_positive(rab);
                    const Tvec n_ij     = dr * rab_inv;

                    // Project velocities onto pair axis
                    const Tscal v_x_a = sycl::dot(vxyz_a, n_ij);
                    const Tscal v_x_b = sycl::dot(vxyz_b, n_ij);

                    // Tangent velocities
                    const Tvec v_t_vec_a  = vxyz_a - n_ij * v_x_a;
                    const Tvec v_t_vec_b  = vxyz_b - n_ij * v_x_b;
                    const Tscal v_t_mag_a = sycl::sqrt(sycl::dot(v_t_vec_a, v_t_vec_a));
                    const Tscal v_t_mag_b = sycl::sqrt(sycl::dot(v_t_vec_b, v_t_vec_b));

                    // Solve exact SR Riemann problem
                    // Riemann solver uses REST-FRAME density n (not lab-frame N)
                    // Convention: Left = b (neighbor), Right = a (current)
                    Tscal P_star, v_x_star, v_t_star;
                    sr::sr_exact_interface(
                        v_x_b,
                        v_t_mag_b,
                        n_b,
                        P_b, // Left state (rest-frame n)
                        v_x_a,
                        v_t_mag_a,
                        n_a,
                        P_a, // Right state (rest-frame n)
                        gamma_eos,
                        P_star,
                        v_x_star,
                        v_t_star);

                    // Kernel gradients (pointing from b to a, i.e., along dr)
                    const Tscal dW_a    = Kernel::dW_3d(rab, h_a);
                    const Tscal dW_b    = Kernel::dW_3d(rab, h_b);
                    const Tvec grad_W_a = n_ij * dW_a;
                    const Tvec grad_W_b = n_ij * dW_b;

                    // SR force computation (Kitajima Eq. 58-59)
                    // dS/dt = -P* V²ij (∇W_i + ∇W_j)
                    // de/dt = v* · dS/dt
                    Tvec dS_contrib;
                    Tscal de_contrib;
                    sr::sr_pairwise_force_simple<Tscal, Tvec>(
                        P_star,
                        v_x_star,
                        n_ij,
                        V_a,
                        V_b,
                        grad_W_a,
                        grad_W_b,
                        dS_contrib,
                        de_contrib);

                    sum_dS += dS_contrib;
                    sum_de += de_contrib;
                });

                // Normalize by baryon number (pmass = ν)
                // dS/dt (per baryon) = sum_dS / ν
                sum_dS = sum_dS / pmass;
                sum_de = sum_de / pmass;

                // Store dS/dt and de/dt to dedicated SR derivative fields
                // These will be used by SR-specific predictor/corrector to integrate
                // S and e directly (not v and u)
                dS_acc[id_a] = sum_dS;
                de_acc[id_a] = sum_de;
            });
        });

        buf_xyz.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_density.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);
        buf_dS.complete_event_state(e);
        buf_de.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

// Explicit template instantiations
// M-spline kernels (Monaghan)
using namespace shammath;
template class shammodels::gsph::modules::UpdateDerivs<f64_3, M4>;
template class shammodels::gsph::modules::UpdateDerivs<f64_3, M6>;
template class shammodels::gsph::modules::UpdateDerivs<f64_3, M8>;

// Wendland kernels (C2, C4, C6)
template class shammodels::gsph::modules::UpdateDerivs<f64_3, C2>;
template class shammodels::gsph::modules::UpdateDerivs<f64_3, C4>;
template class shammodels::gsph::modules::UpdateDerivs<f64_3, C6>;
