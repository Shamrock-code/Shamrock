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
 * @author Guo (guo.yansong@optimind.tech)
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
#include "shammodels/sph/math/density.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::gsph {

    /**
     * @brief Iterative Riemann solver (van Leer 1997)
     *
     * Solves the 1D Riemann problem for two states across an interface.
     * Uses Newton-Raphson iteration to find the contact pressure p*.
     *
     * This implementation matches the reference sphcode's iterative_solver()
     * which uses Lagrangian sound speeds (mass flux) rather than Eulerian.
     *
     * @param rho_L Left density
     * @param u_L Left velocity (projected onto interface normal)
     * @param P_L Left pressure
     * @param rho_R Right density
     * @param u_R Right velocity (projected onto interface normal)
     * @param P_R Right pressure
     * @param gamma Adiabatic index
     * @param tol Convergence tolerance
     * @param max_iter Maximum iterations
     * @return {p_star, v_star} Contact pressure and velocity
     */
    template<class Tscal>
    inline std::pair<Tscal, Tscal> riemann_iterative(
        Tscal rho_L,
        Tscal u_L,
        Tscal P_L,
        Tscal rho_R,
        Tscal u_R,
        Tscal P_R,
        Tscal gamma,
        Tscal tol,
        u32 max_iter) {

        // Input validation
        const Tscal smallp = Tscal(1e-25);
        if (rho_L <= Tscal(0) || rho_R <= Tscal(0) || P_L <= Tscal(0) || P_R <= Tscal(0)) {
            return {Tscal(0.5) * (P_L + P_R), Tscal(0.5) * (u_L + u_R)};
        }

        // Constants (matching reference code)
        const Tscal gamma2 = Tscal(1) + gamma;            // gamma + 1
        const Tscal gamma1 = Tscal(0.5) * gamma2 / gamma; // (gamma + 1) / (2 * gamma)

        // Specific volumes
        const Tscal V_L = Tscal(1) / rho_L;
        const Tscal V_R = Tscal(1) / rho_R;

        // Lagrangian sound speeds (mass flux): c_L = sqrt(gamma * P * rho) = rho * c_s
        const Tscal cl = sycl::sqrt(gamma * P_L * rho_L);
        const Tscal cr = sycl::sqrt(gamma * P_R * rho_R);

        // Initial guess for pstar (from reference code)
        Tscal pstar_guess = P_R - P_L - cr * (u_R - u_L);
        pstar_guess       = P_L + pstar_guess * cl / (cl + cr);
        pstar_guess       = sycl::max(pstar_guess, smallp);

        // Newton-Raphson iteration
        Tscal pstar = pstar_guess;

        for (u32 iter = 0; iter < max_iter; iter++) {
            const Tscal pstar_old = pstar;

            // Left wave impedance
            Tscal w_L = Tscal(1) + gamma1 * (pstar - P_L) / P_L;
            w_L       = cl * sycl::sqrt(sycl::max(w_L, Tscal(0.01)));

            // Right wave impedance
            Tscal w_R = Tscal(1) + gamma1 * (pstar - P_R) / P_R;
            w_R       = cr * sycl::sqrt(sycl::max(w_R, Tscal(0.01)));

            // Left derivative
            Tscal z_L     = Tscal(4) * V_L * w_L * w_L;
            Tscal denom_L = z_L - gamma2 * (pstar - P_L);
            z_L           = -z_L * w_L / (denom_L + Tscal(1e-30));

            // Right derivative
            Tscal z_R     = Tscal(4) * V_R * w_R * w_R;
            Tscal denom_R = z_R - gamma2 * (pstar - P_R);
            z_R           = z_R * w_R / (denom_R + Tscal(1e-30));

            // Intermediate velocities
            const Tscal ustar_L = u_L - (pstar - P_L) / w_L;
            const Tscal ustar_R = u_R + (pstar - P_R) / w_R;

            // Newton-Raphson update
            Tscal denom = z_R - z_L;
            if (sycl::fabs(denom) < Tscal(1e-30)) {
                break; // Avoid division by zero
            }
            pstar = pstar + (ustar_R - ustar_L) * (z_L * z_R) / denom;
            pstar = sycl::max(smallp, pstar);

            // Check convergence
            if (sycl::fabs(pstar - pstar_old) < tol * pstar) {
                break;
            }
        }

        // Recalculate wave impedances with final pstar
        Tscal w_L = Tscal(1) + gamma1 * (pstar - P_L) / P_L;
        w_L       = cl * sycl::sqrt(sycl::max(w_L, Tscal(0.01)));

        Tscal w_R = Tscal(1) + gamma1 * (pstar - P_R) / P_R;
        w_R       = cr * sycl::sqrt(sycl::max(w_R, Tscal(0.01)));

        // Calculate averaged ustar
        const Tscal ustar_L = u_L - (pstar - P_L) / w_L;
        const Tscal ustar_R = u_R + (pstar - P_R) / w_R;
        Tscal ustar         = Tscal(0.5) * (ustar_L + ustar_R);

        // Final validation
        if (!sycl::isfinite(pstar)) {
            pstar = Tscal(0.5) * (P_L + P_R);
        }
        if (!sycl::isfinite(ustar)) {
            ustar = Tscal(0.5) * (u_L + u_R);
        }

        return {pstar, ustar};
    }

    /**
     * @brief HLLC approximate Riemann solver
     *
     * Fast approximate solver using wave speed estimates.
     */
    template<class Tscal>
    inline std::pair<Tscal, Tscal> riemann_hllc(
        Tscal rho_L,
        Tscal u_L,
        Tscal P_L,
        Tscal cs_L,
        Tscal rho_R,
        Tscal u_R,
        Tscal P_R,
        Tscal cs_R) {

        // Input validation
        const Tscal eps = Tscal(1e-30);
        rho_L           = sycl::max(rho_L, eps);
        rho_R           = sycl::max(rho_R, eps);
        P_L             = sycl::max(P_L, eps);
        P_R             = sycl::max(P_R, eps);
        cs_L            = sycl::max(cs_L, Tscal(1e-10));
        cs_R            = sycl::max(cs_R, Tscal(1e-10));

        // Guard against NaN velocities
        if (!sycl::isfinite(u_L))
            u_L = Tscal(0);
        if (!sycl::isfinite(u_R))
            u_R = Tscal(0);

        // PVRS estimate for p*
        Tscal p_pvrs = Tscal(0.5) * (P_L + P_R)
                       - Tscal(0.25) * (u_R - u_L) * (rho_L + rho_R) * (cs_L + cs_R);
        Tscal p_star = sycl::max(p_pvrs, Tscal(0));

        // Wave speed estimates
        Tscal q_L = (p_star > P_L) ? sycl::sqrt(Tscal(1) + Tscal(1.2) * (p_star / P_L - Tscal(1)))
                                   : Tscal(1);
        Tscal q_R = (p_star > P_R) ? sycl::sqrt(Tscal(1) + Tscal(1.2) * (p_star / P_R - Tscal(1)))
                                   : Tscal(1);

        Tscal S_L = u_L - cs_L * q_L;
        Tscal S_R = u_R + cs_R * q_R;

        // Contact wave speed
        Tscal S_star = (P_R - P_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                       / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + Tscal(1e-30));

        // HLLC pressure
        Tscal p_hllc
            = Tscal(0.5) * (P_L + P_R)
              + Tscal(0.5)
                    * (rho_L * (S_L - u_L) * (S_star - u_L) + rho_R * (S_R - u_R) * (S_star - u_R));
        p_hllc = sycl::max(p_hllc, Tscal(1e-10));

        // Final validation
        if (!sycl::isfinite(p_hllc)) {
            p_hllc = Tscal(0.5) * (P_L + P_R);
        }
        if (!sycl::isfinite(S_star)) {
            S_star = Tscal(0.5) * (u_L + u_R);
        }

        return {p_hllc, S_star};
    }

} // namespace shammodels::gsph

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs() {
    StackEntry stack_loc{};

    Cfg_Riemann cfg_riemann = solver_config.riemann_config;

    if (Iterative *v = std::get_if<Iterative>(&cfg_riemann.config)) {
        update_derivs_iterative(*v);
    } else if (HLLC *v = std::get_if<HLLC>(&cfg_riemann.config)) {
        update_derivs_hllc(*v);
    } else if (Exact *v = std::get_if<Exact>(&cfg_riemann.config)) {
        shambase::throw_unimplemented("Exact Riemann solver not yet implemented");
    } else if (Roe *v = std::get_if<Roe>(&cfg_riemann.config)) {
        shambase::throw_unimplemented("Roe Riemann solver not yet implemented");
    } else {
        shambase::throw_unimplemented("Unknown Riemann solver type");
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

    // CRITICAL: Get pressure and soundspeed from storage (includes ghosts after
    // compute_eos_fields!)
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
        // CRITICAL: Use pressure and soundspeed from storage (sized for local + ghost!)
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
        auto xyz         = buf_xyz.get_read_access(depends_list);
        auto axyz        = buf_axyz.get_write_access(depends_list);
        auto vxyz        = buf_vxyz.get_read_access(depends_list);
        auto hpart       = buf_hpart.get_read_access(depends_list);
        auto omega_acc   = buf_omega.get_read_access(depends_list);
        auto density_acc = buf_density.get_read_access(depends_list);
        // Use pressure and soundspeed from storage (includes ghosts!)
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
            const Tscal gamma    = solver_config.gamma;
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

                // Use SPH-summation density (from compute_omega, communicated to ghosts)
                const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));

                // Use pressure and soundspeed from storage (already computed for all particles
                // including ghosts)
                const Tscal P_a  = sycl::max(pressure_acc[id_a], Tscal(1e-30));
                const Tscal cs_a = sycl::max(cs_acc[id_a], Tscal(1e-10));

                // Loop over neighbors using shamrock's neighbor cache
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

                    // Use SPH-summation density (from compute_omega, communicated to ghosts)
                    const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));

                    // Use pressure from storage (includes ghosts!)
                    const Tscal P_b = sycl::max(pressure_acc[id_b], Tscal(1e-30));

                    // Unit vector from a to b (handles rab = 0 gracefully)
                    const Tscal rab_inv  = sham::inv_sat_positive(rab);
                    const Tvec r_ab_unit = dr * rab_inv;

                    // Project velocities onto pair axis for 1D Riemann problem
                    const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                    const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                    // Solve 1D Riemann problem using iterative solver
                    // IMPORTANT: Convention follows reference (g_fluid_force.cpp):
                    //   - r_ab_unit points from b to a (neighbor to current)
                    //   - Along this axis, b is at "left" (lower s), a is at "right" (higher s)
                    //   - Left state = neighbor b, Right state = current a
                    auto [p_star, v_star] = gsph::riemann_iterative<Tscal>(
                        rho_b,
                        u_b_proj,
                        P_b, // Left = neighbor (at lower s along r_ab_unit)
                        rho_a,
                        u_a_proj,
                        P_a, // Right = current (at higher s along r_ab_unit)
                        gamma,
                        tol,
                        max_iter);

                    // Limit p_star to prevent excessive shock forces
                    // Maximum p_star is limited to a multiple of the average pressure
                    const Tscal p_avg      = Tscal(0.5) * (P_a + P_b);
                    const Tscal p_star_max = Tscal(10.0) * sycl::max(p_avg, sycl::max(P_a, P_b));
                    p_star                 = sycl::min(p_star, p_star_max);

                    // Kernel gradients
                    const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // GSPH momentum equation (Cha & Whitworth 2003)
                    // f = dW_a * (m_b * p* / (rho_a^2 * omega_a)) + dW_b * (m_b * p* / (rho_b^2 *
                    // omega_b)) dv_a/dt = -sum_b f
                    const Tscal rho_a_sq = rho_a * rho_a;
                    const Tscal rho_b_sq = rho_b * rho_b;
                    const Tscal coeff
                        = pmass * p_star
                          * (Fab_a / (omega_a * rho_a_sq) + Fab_b / (omega_b * rho_b_sq));

                    sum_axyz -= coeff * r_ab_unit;

                    // GSPH energy equation (matching reference g_fluid_force.cpp)
                    // du_a/dt = -sum_b f · (v* - v_a)
                    // where v* = vstar * e_ij (interface velocity in pair direction)
                    // Since f = coeff * e_ij, we have: du/dt = -coeff * (vstar - u_a_proj)
                    if (do_energy) {
                        sum_du_a -= coeff * (v_star - u_a_proj);
                    }
                });

                // Clamp acceleration to prevent numerical blow-up at shock fronts
                // Maximum acceleration is based on expected shock dynamics
                // For Sod tube: max |a| ~ 10 * max(cs^2) / h ~ 10 * 1.4 / 0.004 ~ 3500
                // Use a generous limit to allow shocks while preventing instabilities
                const Tscal max_acc = Tscal(1e6); // Large but finite limit
                Tscal acc_mag       = sycl::sqrt(sycl::dot(sum_axyz, sum_axyz));
                if (acc_mag > max_acc) {
                    sum_axyz *= max_acc / acc_mag;
                }

                // Clamp du/dt to prevent energy blow-up
                if (do_energy) {
                    const Tscal max_dudt = Tscal(1e6);
                    sum_du_a             = sycl::clamp(sum_du_a, -max_dudt, max_dudt);
                }

                // Write accumulated derivatives
                axyz[id_a] = sum_axyz;
                if (do_energy && duint_acc != nullptr) {
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

    // CRITICAL: Get pressure and soundspeed from storage (includes ghosts after
    // compute_eos_fields!)
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
        // CRITICAL: Use pressure and soundspeed from storage (sized for local + ghost!)
        sham::DeviceBuffer<Tscal> &buf_pressure
            = pressure_field.get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = soundspeed_field.get_field(cur_p.id_patch).get_buf();

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        // Get density from merged ghost data (SPH summation density)
        sham::DeviceBuffer<Tscal> &buf_density = mpdat.get_field_buf_ref<Tscal>(idensity_interf);

        auto xyz         = buf_xyz.get_read_access(depends_list);
        auto axyz        = buf_axyz.get_write_access(depends_list);
        auto vxyz        = buf_vxyz.get_read_access(depends_list);
        auto hpart       = buf_hpart.get_read_access(depends_list);
        auto omega_acc   = buf_omega.get_read_access(depends_list);
        auto density_acc = buf_density.get_read_access(depends_list);
        // Use pressure and soundspeed from storage (includes ghosts!)
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
            const Tscal gamma    = solver_config.gamma;
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

                // Use SPH-summation density (from compute_omega, communicated to ghosts)
                const Tscal rho_a = sycl::max(density_acc[id_a], Tscal(1e-30));

                // Use pressure and soundspeed from storage (already computed for all particles
                // including ghosts)
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

                    // Use SPH-summation density (from compute_omega, communicated to ghosts)
                    const Tscal rho_b = sycl::max(density_acc[id_b], Tscal(1e-30));

                    // Use pressure and soundspeed from storage (includes ghosts!)
                    const Tscal P_b  = sycl::max(pressure_acc[id_b], Tscal(1e-30));
                    const Tscal cs_b = sycl::max(cs_acc[id_b], Tscal(1e-10));

                    const Tscal rab_inv  = sham::inv_sat_positive(rab);
                    const Tvec r_ab_unit = dr * rab_inv;

                    const Tscal u_a_proj = sycl::dot(vxyz_a, r_ab_unit);
                    const Tscal u_b_proj = sycl::dot(vxyz_b, r_ab_unit);

                    // Use HLLC approximate Riemann solver (faster than iterative)
                    // IMPORTANT: Convention follows reference (g_fluid_force.cpp):
                    //   - Left state = neighbor b, Right state = current a
                    auto [p_star, v_star] = gsph::riemann_hllc<Tscal>(
                        rho_b,
                        u_b_proj,
                        P_b,
                        cs_b, // Left = neighbor
                        rho_a,
                        u_a_proj,
                        P_a,
                        cs_a); // Right = current

                    // Limit p_star to prevent excessive shock forces
                    const Tscal p_avg      = Tscal(0.5) * (P_a + P_b);
                    const Tscal p_star_max = Tscal(10.0) * sycl::max(p_avg, sycl::max(P_a, P_b));
                    p_star                 = sycl::min(p_star, p_star_max);

                    const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    const Tscal rho_a_sq = rho_a * rho_a;
                    const Tscal rho_b_sq = rho_b * rho_b;
                    const Tscal coeff
                        = pmass * p_star
                          * (Fab_a / (omega_a * rho_a_sq) + Fab_b / (omega_b * rho_b_sq));

                    sum_axyz -= coeff * r_ab_unit;

                    // GSPH energy equation (matching reference g_fluid_force.cpp)
                    // du_a/dt = -sum_b f · (v* - v_a)
                    if (do_energy) {
                        sum_du_a -= coeff * (v_star - u_a_proj);
                    }
                });

                // Clamp acceleration to prevent numerical blow-up at shock fronts
                const Tscal max_acc = Tscal(1e6);
                Tscal acc_mag       = sycl::sqrt(sycl::dot(sum_axyz, sum_axyz));
                if (acc_mag > max_acc) {
                    sum_axyz *= max_acc / acc_mag;
                }

                // Clamp du/dt to prevent energy blow-up
                if (do_energy) {
                    const Tscal max_dudt = Tscal(1e6);
                    sum_du_a             = sycl::clamp(sum_du_a, -max_dudt, max_dudt);
                }

                axyz[id_a] = sum_axyz;
                if (do_energy && duint_acc != nullptr) {
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
