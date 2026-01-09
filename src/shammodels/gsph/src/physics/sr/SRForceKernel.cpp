// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SRForceKernel.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Implementation of Special Relativistic GSPH force kernel
 *
 * Implements force computation for SR hydrodynamics following
 * Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 */

#include "shambase/exception.hpp"
#include "shambackends/math.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/physics/sr/SRFieldNames.hpp"
#include "shammodels/gsph/physics/sr/SRForceKernel.hpp"
#include "shammodels/gsph/physics/sr/forces.hpp"
#include "shammodels/gsph/physics/sr/riemann/Exact.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::gsph::physics::sr {

    template<class Tvec, template<class> class SPHKernel>
    void SRForceKernel<Tvec, SPHKernel>::compute_exact(const riemann::ExactConfig &cfg) {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        using Kernel = SPHKernel<Tscal>;

        // Get ghost layout for reading particle data (ghosts included)
        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage_.ghost_layout.get());
        u32 ihpart_interf      = ghost_layout.get_field_idx<Tscal>(fields::HPART);
        u32 ivxyz_interf       = ghost_layout.get_field_idx<Tvec>(fields::VXYZ);
        u32 iN_labframe_interf = ghost_layout.get_field_idx<Tscal>(fields::N_LABFRAME);
        u32 iomega_interf      = ghost_layout.get_field_idx<Tscal>(fields::OMEGA);
        u32 iuint_interf       = ghost_layout.get_field_idx<Tscal>(fields::UINT);

        // Per-particle baryon number (ν) for volume-based h (Kitajima)
        const bool has_pmass = config_.has_field_pmass();
        u32 ipmass_interf    = has_pmass ? ghost_layout.get_field_idx<Tscal>("pmass") : 0;

        auto &merged_xyzh                                 = storage_.merged_xyzh.get();
        shambase::DistributedData<PatchDataLayer> &mpdats = storage_.merged_patchdata_ghost.get();

        // Get SR conserved variable derivative fields from storage
        shamrock::solvergraph::Field<Tvec> &dS_field
            = shambase::get_check_ref(storage_.dS_momentum);
        shamrock::solvergraph::Field<Tscal> &de_field = shambase::get_check_ref(storage_.de_energy);

        // SR config parameters
        const Tscal c_speed   = config_.c_speed;
        const Tscal gamma_eos = config_.get_eos_gamma();
        const Tscal pmass     = config_.gpart_mass;
        const bool use_grad_h = config_.use_grad_h;
        const Tscal hfact     = SPHKernel<Tscal>::hfactd;

        scheduler_.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

            sham::DeviceBuffer<Tvec> &buf_xyz
                = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
            sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
            sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
            sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);

            // Get SR derivative buffers from storage fields
            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_field(cur_p.id_patch).get_buf();
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_field(cur_p.id_patch).get_buf();

            tree::ObjectCache &pcache
                = shambase::get_check_ref(storage_.neigh_cache).get_cache(cur_p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            // Lab-frame baryon density N (from kernel summation)
            sham::DeviceBuffer<Tscal> &buf_N = mpdat.get_field_buf_ref<Tscal>(iN_labframe_interf);

            // Per-particle baryon number buffer (only when volume-based h enabled)
            sham::DeviceBuffer<Tscal> *buf_pmass_ptr = nullptr;
            if (has_pmass) {
                buf_pmass_ptr = &mpdat.get_field_buf_ref<Tscal>(ipmass_interf);
            }

            auto xyz_labframe_acc = buf_xyz.get_read_access(depends_list);
            auto v_labframe_acc   = buf_vxyz.get_read_access(depends_list);
            auto hpart_acc        = buf_hpart.get_read_access(depends_list);
            auto omega_acc        = buf_omega.get_read_access(depends_list);
            auto N_labframe_acc   = buf_N.get_read_access(depends_list);
            auto u_restframe_acc  = buf_uint.get_read_access(depends_list);
            auto ploop_ptrs       = pcache.get_read_access(depends_list);

            // Per-particle pmass accessor (nullptr if not using volume-based h)
            const Tscal *pmass_acc
                = has_pmass ? buf_pmass_ptr->get_read_access(depends_list) : nullptr;

            // Write access for SR derivative fields
            auto dS_acc = buf_dS.get_write_access(depends_list);
            auto de_acc = buf_de.get_write_access(depends_list);

            // Number of real particles in this patch
            const u32 n_real = pdat.get_obj_cnt();

            auto e = q.submit(
                depends_list, [&, use_grad_h, has_pmass, hfact, n_real](sycl::handler &cgh) {
                    const Tscal c2 = c_speed * c_speed;

                    tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    shambase::parallel_for(
                        cgh, pdat.get_obj_cnt(), "SR-GSPH derivs exact", [=](u64 gid) {
                            u32 id_a = (u32) gid;

                            Tvec sum_dS  = {0, 0, 0}; // dS/dt accumulator (momentum)
                            Tscal sum_de = 0;         // de/dt accumulator (energy)

                            const Tscal h_a           = hpart_acc[id_a];
                            const Tvec xyz_labframe_a = xyz_labframe_acc[id_a];
                            const Tvec v_labframe_a   = v_labframe_acc[id_a];
                            const Tscal omega_a       = omega_acc[id_a];

                            // Per-particle baryon number (νᵢ) for Kitajima, or global pmass
                            const Tscal nu_a = has_pmass ? pmass_acc[id_a] : pmass;

                            // Lab-frame baryon density N (from kernel summation)
                            const Tscal N_labframe_a = N_labframe_acc[id_a];

                            // Compute Lorentz factor
                            const Tscal v2_a = sycl::dot(v_labframe_a, v_labframe_a) / c2;
                            const Tscal gamma_a
                                = Tscal{1} / sycl::sqrt(sycl::fmax(Tscal{1} - v2_a, Tscal{1e-10}));

                            // Rest-frame density: n = N/γ
                            const Tscal n_restframe_a = N_labframe_a / gamma_a;

                            // Rest-frame internal energy and pressure
                            const Tscal u_restframe_a = u_restframe_acc[id_a];
                            const Tscal P_restframe_a
                                = (gamma_eos - Tscal{1}) * n_restframe_a * u_restframe_a;

                            // Specific enthalpy: H = 1 + u/c² + P/(nc²) (rest-frame quantities)
                            const Tscal H_a = Tscal{1} + u_restframe_a / c2
                                              + P_restframe_a / (n_restframe_a * c2);

                            // Particle volume: V_p = h³/hfact³ (independent of ν!)
                            // Kitajima Eq. 221: V_p = 1/W_sum ≈ h³/hfact³
                            const Tscal h_a_hfact = h_a / hfact;
                            const Tscal V_a       = h_a_hfact * h_a_hfact * h_a_hfact;

                            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                                if (id_a == id_b)
                                    return;

                                const Tvec dr    = xyz_labframe_a - xyz_labframe_acc[id_b];
                                const Tscal rab2 = sycl::dot(dr, dr);
                                const Tscal h_b  = hpart_acc[id_b];

                                if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                                    return;
                                }

                                const Tscal rab         = sycl::sqrt(rab2);
                                const Tvec v_labframe_b = v_labframe_acc[id_b];
                                const Tscal omega_b     = omega_acc[id_b];

                                // Lab-frame baryon density N (from kernel summation)
                                const Tscal N_labframe_b = N_labframe_acc[id_b];

                                // Relativistic quantities for particle b
                                const Tscal v2_b = sycl::dot(v_labframe_b, v_labframe_b) / c2;
                                const Tscal gamma_b
                                    = Tscal{1}
                                      / sycl::sqrt(sycl::fmax(Tscal{1} - v2_b, Tscal{1e-10}));

                                // Rest-frame density: n = N/γ
                                const Tscal n_restframe_b = N_labframe_b / gamma_b;

                                // Rest-frame internal energy and pressure for particle b
                                const Tscal u_restframe_b = u_restframe_acc[id_b];
                                const Tscal P_restframe_b
                                    = (gamma_eos - Tscal{1}) * n_restframe_b * u_restframe_b;

                                const Tscal H_b = Tscal{1} + u_restframe_b / c2
                                                  + P_restframe_b / (n_restframe_b * c2);

                                // Particle volume: V_p = h³/hfact³ (independent of ν!)
                                const Tscal h_b_hfact = h_b / hfact;
                                const Tscal V_b       = h_b_hfact * h_b_hfact * h_b_hfact;

                                // Unit vector from a to b (matching reference convention)
                                // Reference: n_ij = -r_ij/|r_ij| where r_ij = pos[i] - pos[j]
                                // So n_ij points from current (i) to neighbor (j)
                                const Tscal rab_inv = sham::inv_sat_positive(rab);
                                const Tvec n_ij     = -dr * rab_inv; // Negate to point from a to b

                                // Project velocities onto pair axis
                                const Tscal v_x_a = sycl::dot(v_labframe_a, n_ij);
                                const Tscal v_x_b = sycl::dot(v_labframe_b, n_ij);

                                // Tangent velocities
                                const Tvec v_t_vec_a  = v_labframe_a - n_ij * v_x_a;
                                const Tvec v_t_vec_b  = v_labframe_b - n_ij * v_x_b;
                                const Tscal v_t_mag_a = sycl::sqrt(sycl::dot(v_t_vec_a, v_t_vec_a));
                                const Tscal v_t_mag_b = sycl::sqrt(sycl::dot(v_t_vec_b, v_t_vec_b));

                                // Solve exact SR Riemann problem
                                // Riemann solver uses REST-FRAME density n (not lab-frame N)
                                // Convention: Left = a (current), Right = b (neighbor)
                                // This matches reference where n_ij points from Left to Right
                                auto riemann_result
                                    = ::shammodels::gsph::physics::sr::riemann::solve(
                                        v_x_a,
                                        v_t_mag_a,
                                        n_restframe_a,
                                        P_restframe_a,
                                        v_x_b,
                                        v_t_mag_b,
                                        n_restframe_b,
                                        P_restframe_b,
                                        gamma_eos);

                                // Use Riemann solver output directly
                                Tscal P_star   = riemann_result.P_star;
                                Tscal v_x_star = riemann_result.v_x_star;
                                Tscal v_t_star = riemann_result.v_t_star;

                                // Kernel gradients with √2h (Kitajima Eq. 24, GSPH convolution)
                                // Note: Use -n_ij to maintain gradient convention
                                // ∇_a W = dW * (x_a - x_b)/|r| = dW * (-n_ij) since n_ij points
                                // from a to b
                                constexpr Tscal sqrt2 = Tscal{1.4142135623730951};
                                const Tscal dW_a      = Kernel::dW_3d(rab, sqrt2 * h_a);
                                const Tscal dW_b      = Kernel::dW_3d(rab, sqrt2 * h_b);
                                const Tvec grad_W_a   = -n_ij * dW_a;
                                const Tvec grad_W_b   = -n_ij * dW_b;

                                // Tangent velocity unit vectors
                                const Tscal v_t_mag_a_safe = sycl::fmax(v_t_mag_a, Tscal{1e-30});
                                const Tscal v_t_mag_b_safe = sycl::fmax(v_t_mag_b, Tscal{1e-30});
                                const Tvec v_t_dir_a       = v_t_vec_a / v_t_mag_a_safe;
                                const Tvec v_t_dir_b       = v_t_vec_b / v_t_mag_b_safe;

                                // SR force computation (Kitajima Eq. 371-374)
                                // Pass V_a and V_b for V²_interp = (V_a² + V_b²)/2
                                // v_t_dir_L = v_t_dir_a (Left = a), v_t_dir_R = v_t_dir_b (Right =
                                // b)
                                Tvec dS_contrib;
                                Tscal de_contrib;
                                ::shammodels::gsph::physics::sr::sr_pairwise_force<Tscal, Tvec>(
                                    P_star,
                                    v_x_star,
                                    v_t_star,
                                    n_ij,
                                    v_t_dir_a,
                                    v_t_dir_b,
                                    V_a,
                                    V_b,
                                    grad_W_a,
                                    grad_W_b,
                                    dS_contrib,
                                    de_contrib);

                                // Kitajima Eq. 371: <νᵢ dSᵢ/dt> = -Σⱼ P* V² [∇ᵢW - ∇ⱼW]
                                // NOTE: No νⱼ factor in the sum! This is per-baryon rate.
                                sum_dS += dS_contrib;
                                sum_de += de_contrib;
                            });

                            // Kitajima Eq. 371: νᵢṠᵢ = -Σⱼ P* V² [∇ᵢW - ∇ⱼW]
                            // Divide by νᵢ (per-particle baryon number) to get per-baryon momentum
                            // rate:
                            //   Ṡᵢ = (-Σⱼ P* V² ∇W) / νᵢ

                            // Store dS/dt and de/dt (normalized by per-particle baryon number)
                            const Tscal nu_a_safe = sycl::fmax(nu_a, Tscal{1e-30});
                            dS_acc[id_a]          = sum_dS / nu_a_safe;
                            de_acc[id_a]          = sum_de / nu_a_safe;
                        });
                });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_N.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_dS.complete_event_state(e);
            buf_de.complete_event_state(e);
            if (has_pmass && buf_pmass_ptr) {
                buf_pmass_ptr->complete_event_state(e);
            }

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRForceKernel<Tvec, SPHKernel>::compute_hllc(const riemann::HLLC_SRConfig &cfg) {
        shambase::throw_unimplemented("HLLC-SR Riemann solver not yet implemented");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Explicit template instantiations
    // ════════════════════════════════════════════════════════════════════════════

    using namespace shammath;

    template class SRForceKernel<sycl::vec<double, 3>, M4>;
    template class SRForceKernel<sycl::vec<double, 3>, M6>;
    template class SRForceKernel<sycl::vec<double, 3>, M8>;
    template class SRForceKernel<sycl::vec<double, 3>, C2>;
    template class SRForceKernel<sycl::vec<double, 3>, C4>;
    template class SRForceKernel<sycl::vec<double, 3>, C6>;
    template class SRForceKernel<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::sr
