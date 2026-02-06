// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file UpdateDerivs.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/mhd.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shammodels/sph/modules/UpdateDerivs.hpp"
#include "shamphys/mhd.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs() {

    Cfg_AV cfg_av   = solver_config.artif_viscosity;
    Cfg_MHD cfg_mhd = solver_config.mhd_config;

    if (Constant *v = std::get_if<Constant>(&cfg_av.config)) {
        update_derivs_constantAV(*v);
    } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&cfg_av.config)) {
        update_derivs_mm97(*v);
    } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&cfg_av.config)) {
        update_derivs_cd10(*v);
    } else if (ConstantDisc *v = std::get_if<ConstantDisc>(&cfg_av.config)) {
        update_derivs_disc_visco(*v);
    } else if (IdealMHD *v = std::get_if<IdealMHD>(&cfg_mhd.config)) {
        update_derivs_MHD(*v);
    } else if (NonIdealMHD *v = std::get_if<NonIdealMHD>(&cfg_mhd.config)) {
        shambase::throw_unimplemented();
    } else if (NoneMHD *v = std::get_if<NoneMHD>(&cfg_mhd.config)) {
        shambase::throw_unimplemented();
    } else if (None *v = std::get_if<None>(&cfg_av.config)) {
        shambase::throw_unimplemented();
    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_noAV(None cfg) {}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_constantAV(
    Constant cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega        = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = shambase::get_check_ref(storage.pressure).get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs
            = shambase::get_check_ref(storage.soundspeed).get_field(cur_p.id_patch).get_buf();

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz        = buf_xyz.get_read_access(depends_list);
        auto axyz       = buf_axyz.get_write_access(depends_list);
        auto du         = buf_duint.get_write_access(depends_list);
        auto vxyz       = buf_vxyz.get_read_access(depends_list);
        auto hpart      = buf_hpart.get_read_access(depends_list);
        auto omega      = buf_omega.get_read_access(depends_list);
        auto u          = buf_uint.get_read_access(depends_list); // TODO rename to uint
        auto pressure   = buf_pressure.get_read_access(depends_list);
        auto cs         = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal alpha_u  = cfg.alpha_u;
            const Tscal alpha_AV = cfg.alpha_AV;
            const Tscal beta_AV  = cfg.beta_AV;

            shamlog_debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            shamlog_debug_sycl_ln("deriv kernel", "alpha_AV :", alpha_AV);
            shamlog_debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute force cte AV", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tscal cs_a = cs[id_a];

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // compute only omega_a
                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tscal rab       = sycl::sqrt(rab2);
                    Tvec vxyz_b     = vxyz[id_b];
                    const Tscal u_b = u[id_b];

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                    Tscal P_b   = pressure[id_b];
                    // f32 P_b     = cs * cs * rho_b;
                    Tscal omega_b = omega[id_b];
                    Tscal cs_b    = cs[id_b];

                    const Tscal alpha_a = alpha_AV;
                    const Tscal alpha_b = alpha_AV;

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    Tvec v_ab = vxyz_a - vxyz_b;

                    Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                    // f32 P_b     = cs * cs * rho_b;
                    Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
                    Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                    Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
                    Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

                    Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

                    Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
                    Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

                    add_to_derivs_sph_artif_visco_cond(
                        pmass,
                        rho_a_sq,
                        omega_a_rho_a_inv,
                        rho_a_inv,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        u_a,
                        u_b,
                        P_a,
                        P_b,
                        alpha_u,
                        v_ab,
                        r_ab_unit,
                        vsig_u,
                        qa_ab,
                        qb_ab,
                        force_pressure,
                        tmpdU_pressure);
                });
                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });

        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_duint.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_uint.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_mm97(VaryingMM97 cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");
    const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf    = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf     = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf     = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf    = ghost_layout.get_field_idx<Tscal>("omega");
    u32 ialpha_AV_interf = ghost_layout.get_field_idx<Tscal>("alpha_AV");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega        = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = shambase::get_check_ref(storage.pressure).get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_alpha_AV
            = storage.alpha_av_ghost.get().get(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs
            = shambase::get_check_ref(storage.soundspeed).get_field(cur_p.id_patch).get_buf();

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz        = buf_xyz.get_read_access(depends_list);
        auto axyz       = buf_axyz.get_write_access(depends_list);
        auto du         = buf_duint.get_write_access(depends_list);
        auto vxyz       = buf_vxyz.get_read_access(depends_list);
        auto hpart      = buf_hpart.get_read_access(depends_list);
        auto omega      = buf_omega.get_read_access(depends_list);
        auto u          = buf_uint.get_read_access(depends_list);
        auto pressure   = buf_pressure.get_read_access(depends_list);
        auto alpha_AV   = buf_alpha_AV.get_read_access(depends_list);
        auto cs         = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass   = solver_config.gpart_mass;
            const Tscal alpha_u = cfg.alpha_u;
            const Tscal beta_AV = cfg.beta_AV;

            shamlog_debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            shamlog_debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute force MM97 AV", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tscal cs_a = cs[id_a];

                const Tscal alpha_a = alpha_AV[id_a];

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // compute only omega_a
                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tscal rab       = sycl::sqrt(rab2);
                    Tvec vxyz_b     = vxyz[id_b];
                    const Tscal u_b = u[id_b];

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                    Tscal P_b   = pressure[id_b];
                    // f32 P_b     = cs * cs * rho_b;
                    Tscal omega_b = omega[id_b];
                    Tscal cs_b    = cs[id_b];

                    /////////////////
                    // internal energy update
                    //  scalar : f32  | vector : f32_3
                    const Tscal alpha_b = alpha_AV[id_b];

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    Tvec v_ab = vxyz_a - vxyz_b;

                    Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                    // f32 P_b     = cs * cs * rho_b;
                    Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
                    Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                    Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
                    Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

                    Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

                    Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
                    Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

                    add_to_derivs_sph_artif_visco_cond(
                        pmass,
                        rho_a_sq,
                        omega_a_rho_a_inv,
                        rho_a_inv,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        u_a,
                        u_b,
                        P_a,
                        P_b,
                        alpha_u,
                        v_ab,
                        r_ab_unit,
                        vsig_u,
                        qa_ab,
                        qb_ab,

                        force_pressure,
                        tmpdU_pressure);
                });

                // sum_du_a               = P_a * rho_a_inv * omega_a_rho_a_inv * sum_du_a;
                // lambda_viscous_heating = -omega_a_rho_a_inv * lambda_viscous_heating;
                // lambda_shock           = lambda_viscous_heating + lambda_conductivity;
                // sum_du_a               = sum_du_a + lambda_shock;

                // out << "sum : " << sum_axyz << "\n";

                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });

        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_duint.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_uint.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_alpha_AV.complete_event_state(e);
        buf_cs.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_cd10(VaryingCD10 cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega        = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    auto &part_counts            = storage.part_counts;
    auto &part_counts_with_ghost = storage.part_counts_with_ghost;
    auto &xyz_refs               = storage.positions_with_ghosts;
    auto &omega_field            = storage.omega;
    auto &pressure_field         = storage.pressure;
    auto &soundspeed_field       = storage.soundspeed;

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> uint_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");

    { // Attach spans to block coords
        shambase::get_check_ref(uint_refs).set_refs(
            mpdats.map<std::reference_wrapper<PatchDataField<Tscal>>>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tscal>(iuint_interf));
                }));
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> vxyz_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("", "");
    { // Attach spans to block coords
        shambase::get_check_ref(vxyz_refs).set_refs(
            mpdats.map<std::reference_wrapper<PatchDataField<Tvec>>>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tvec>(ivxyz_interf));
                }));
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    { // if was just reset before this call
        shambase::get_check_ref(hpart_refs)
            .set_refs(mpdats.map<std::reference_wrapper<PatchDataField<Tscal>>>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tscal>(ihpart_interf));
                }));
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> omega_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    { // Attach spans to block coords
        shambase::get_check_ref(omega_refs)
            .set_refs(mpdats.map<std::reference_wrapper<PatchDataField<Tscal>>>(
                [&](u64 id, shamrock::patch::PatchDataLayer &mpdat) {
                    return std::ref(mpdat.get_field<Tscal>(iomega_interf));
                }));
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> alpha_av_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    { // Attach spans to block coords
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>> refs{};
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            refs.add_obj(
                cur_p.id_patch, std::ref(storage.alpha_av_ghost.get().get(cur_p.id_patch)));
        });
        shambase::get_check_ref(alpha_av_refs).set_refs(refs);
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> axyz_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("", "");
    { // Attach spans to block coords
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tvec>>> refs{};
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            refs.add_obj(cur_p.id_patch, std::ref(pdat.get_field<Tvec>(iaxyz)));
        });
        shambase::get_check_ref(axyz_refs).set_refs(refs);
    }

    std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> duint_refs
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("", "");
    { // Attach spans to block coords
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>> refs{};
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            refs.add_obj(cur_p.id_patch, std::ref(pdat.get_field<Tscal>(iduint)));
        });
        shambase::get_check_ref(duint_refs).set_refs(refs);
    }

    struct Edges {
        // in counts
        const shamrock::solvergraph::Indexes<u32> &part_counts;
        const shamrock::solvergraph::Indexes<u32> &part_counts_with_ghost;

        // in fields
        const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
        const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
        const shamrock::solvergraph::IFieldSpan<Tvec> &vxyz;
        const shamrock::solvergraph::IFieldSpan<Tscal> &uint;
        const shamrock::solvergraph::IFieldSpan<Tscal> &omega;
        const shamrock::solvergraph::IFieldSpan<Tscal> &pressure;
        const shamrock::solvergraph::IFieldSpan<Tscal> &cs;
        const shamrock::solvergraph::IFieldSpan<Tscal> &alpha_AV;

        // in neigh list
        const shammodels::sph::solvergraph::NeighCache &neigh_cache;

        // outputs
        shamrock::solvergraph::IFieldSpan<Tvec> &axyz;
        shamrock::solvergraph::IFieldSpan<Tscal> &duint;
    };

    Edges edges{
        shambase::get_check_ref(part_counts),
        shambase::get_check_ref(part_counts_with_ghost),
        shambase::get_check_ref(xyz_refs),
        shambase::get_check_ref(hpart_refs),
        shambase::get_check_ref(vxyz_refs),
        shambase::get_check_ref(uint_refs),
        shambase::get_check_ref(omega_refs),
        shambase::get_check_ref(pressure_field),
        shambase::get_check_ref(soundspeed_field),
        shambase::get_check_ref(alpha_av_refs),
        shambase::get_check_ref(storage.neigh_cache),
        shambase::get_check_ref(axyz_refs),
        shambase::get_check_ref(duint_refs),
    };

    edges.xyz.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.hpart.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.vxyz.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.uint.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.omega.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.pressure.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.cs.check_sizes(edges.part_counts_with_ghost.indexes);
    edges.alpha_AV.check_sizes(edges.part_counts_with_ghost.indexes);

    edges.axyz.check_sizes(edges.part_counts.indexes);
    edges.duint.check_sizes(edges.part_counts.indexes);

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    const Tscal pmass   = solver_config.gpart_mass;
    const Tscal alpha_u = cfg.alpha_u;
    const Tscal beta_AV = cfg.beta_AV;

    auto kernel = [pmass, alpha_u, beta_AV](
                      u32 id_a,
                      const Tvec *__restrict xyz,
                      const Tscal *__restrict hpart,
                      const Tvec *__restrict vxyz,
                      const Tscal *__restrict uint,
                      const Tscal *__restrict omega,
                      const Tscal *__restrict pressure,
                      const Tscal *__restrict cs,
                      const Tscal *__restrict alpha_AV,
                      const shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
                      // outs
                      Tvec *__restrict axyz,
                      Tscal *__restrict duint) {
        // init utilities
        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;
        tree::ObjectCacheIterator particle_looper(ploop_ptrs);

        using namespace shamrock::sph;

        Tvec xyz_a    = xyz[id_a];
        Tscal h_a     = hpart[id_a];
        Tvec vxyz_a   = vxyz[id_a];
        Tscal u_a     = uint[id_a];
        Tscal omega_a = omega[id_a];
        Tscal P_a     = pressure[id_a];
        Tscal cs_a    = cs[id_a];
        Tscal alpha_a = alpha_AV[id_a];

        Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
        Tscal rho_a_sq  = rho_a * rho_a;
        Tscal rho_a_inv = 1. / rho_a;

        Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

        Tvec force_pressure{0, 0, 0};
        Tscal tmpdU_pressure = 0;
        Tvec sum_axyz        = {0, 0, 0};
        Tscal sum_du_a       = 0;

        particle_looper.for_each_object(id_a, [&](u32 id_b) {
            // compute only omega_a
            Tvec dr    = xyz_a - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);
            Tscal h_b  = hpart[id_b];

            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                return;
            }

            Tvec vxyz_b         = vxyz[id_b];
            const Tscal u_b     = uint[id_b];
            Tscal P_b           = pressure[id_b];
            Tscal omega_b       = omega[id_b];
            const Tscal alpha_b = alpha_AV[id_b];
            Tscal cs_b          = cs[id_b];

            Tscal rab = sycl::sqrt(rab2);

            Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

            Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            Tvec v_ab = vxyz_a - vxyz_b;

            Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

            // f32 P_b     = cs * cs * rho_b;
            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
            Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

            Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
            Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

            Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

            Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
            Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

            add_to_derivs_sph_artif_visco_cond(
                pmass,
                rho_a_sq,
                omega_a_rho_a_inv,
                rho_a_inv,
                rho_b,
                omega_a,
                omega_b,
                Fab_a,
                Fab_b,
                u_a,
                u_b,
                P_a,
                P_b,
                alpha_u,
                v_ab,
                r_ab_unit,
                vsig_u,
                qa_ab,
                qb_ab,

                force_pressure,
                tmpdU_pressure);
        });

        axyz[id_a]  = force_pressure;
        duint[id_a] = tmpdU_pressure;
    };

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.vxyz.get_spans(),
            edges.uint.get_spans(),
            edges.omega.get_spans(),
            edges.pressure.get_spans(),
            edges.cs.get_spans(),
            edges.alpha_AV.get_spans(),
            edges.neigh_cache},
        sham::DDMultiRef{edges.axyz.get_spans(), edges.duint.get_spans()},
        edges.part_counts.indexes,
        std::move(kernel));
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_disc_visco(
    ConstantDisc cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega        = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = shambase::get_check_ref(storage.pressure).get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs
            = shambase::get_check_ref(storage.soundspeed).get_field(cur_p.id_patch).get_buf();

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz        = buf_xyz.get_read_access(depends_list);
        auto axyz       = buf_axyz.get_write_access(depends_list);
        auto du         = buf_duint.get_write_access(depends_list);
        auto vxyz       = buf_vxyz.get_read_access(depends_list);
        auto hpart      = buf_hpart.get_read_access(depends_list);
        auto omega      = buf_omega.get_read_access(depends_list);
        auto u          = buf_uint.get_read_access(depends_list);
        auto pressure   = buf_pressure.get_read_access(depends_list);
        auto cs         = buf_cs.get_read_access(depends_list);
        auto ploop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal alpha_AV = cfg.alpha_AV;
            const Tscal alpha_u  = cfg.alpha_u;
            const Tscal beta_AV  = cfg.beta_AV;

            shamlog_debug_sycl_ln("deriv kernel", "alpha_AV  :", alpha_AV);
            shamlog_debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            shamlog_debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute force disc", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal cs_a      = cs[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                // f32 P_a     = cs * cs * rho_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // compute only omega_a
                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tvec vxyz_b     = vxyz[id_b];
                    const Tscal u_b = u[id_b];
                    Tscal P_b       = pressure[id_b];
                    Tscal omega_b   = omega[id_b];
                    Tscal cs_b      = cs[id_b];

                    Tscal rab = sycl::sqrt(rab2);

                    Tscal rho_b         = rho_h(pmass, h_b, Kernel::hfactd);
                    const Tscal alpha_a = alpha_AV;
                    const Tscal alpha_b = alpha_AV;
                    Tscal Fab_a         = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b         = Kernel::dW_3d(rab, h_b);

                    Tvec v_ab = vxyz_a - vxyz_b;

                    Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                    // f32 P_b     = cs * cs * rho_b;
                    Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
                    Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                    Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
                    Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

                    Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

                    Tscal qa_ab = shamrock::sph::q_av_disc(
                        rho_a, h_a, rab, alpha_a, cs_a, vsig_a, v_ab_r_ab);
                    Tscal qb_ab = shamrock::sph::q_av_disc(
                        rho_b, h_b, rab, alpha_b, cs_b, vsig_b, v_ab_r_ab);

                    add_to_derivs_sph_artif_visco_cond(
                        pmass,
                        rho_a_sq,
                        omega_a_rho_a_inv,
                        rho_a_inv,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        u_a,
                        u_b,
                        P_a,
                        P_b,
                        alpha_u,
                        v_ab,
                        r_ab_unit,
                        vsig_u,
                        qa_ab,
                        qb_ab,

                        force_pressure,
                        tmpdU_pressure);
                });

                axyz[id_a] = force_pressure;
                du[id_a]   = tmpdU_pressure;
            });
        });

        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_duint.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_uint.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_MHD(IdealMHD cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 ixyz        = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz       = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz       = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint       = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint      = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart      = pdl.get_field_idx<Tscal>("hpart");
    const u32 iB_on_rho   = pdl.get_field_idx<Tvec>("B/rho");
    const u32 idB_on_rho  = pdl.get_field_idx<Tvec>("dB/rho");
    const u32 ipsi_on_ch  = pdl.get_field_idx<Tscal>("psi/ch");
    const u32 idpsi_on_ch = pdl.get_field_idx<Tscal>("dpsi/ch");
    const u32 idrho_dt    = pdl.get_field_idx<Tscal>("drho/dt");

    bool do_MHD_debug       = solver_config.do_MHD_debug();
    const u32 imag_pressure = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("mag_pressure") : -1;
    const u32 imag_tension  = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("mag_tension") : -1;
    const u32 igas_pressure = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("gas_pressure") : -1;
    const u32 itensile_corr = (do_MHD_debug) ? pdl.get_field_idx<Tvec>("tensile_corr") : -1;
    const u32 ipsi_propag   = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_propag") : -1;
    const u32 ipsi_diff     = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_diff") : -1;
    const u32 ipsi_cons     = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("psi_cons") : -1;
    const u32 iu_mhd        = (do_MHD_debug) ? pdl.get_field_idx<Tscal>("u_mhd") : -1;

    // Tscal mu_0 = 1.;
    Tscal const mu_0 = solver_config.get_constant_mu_0();

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf     = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf      = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf      = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf     = ghost_layout.get_field_idx<Tscal>("omega");
    u32 iB_on_rho_interf  = ghost_layout.get_field_idx<Tvec>("B/rho");
    u32 ipsi_on_ch_interf = ghost_layout.get_field_idx<Tscal>("psi/ch");

    // logger::raw_ln("charged the ghost fields.");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    shamrock::solvergraph::Field<Tscal> &omega        = shambase::get_check_ref(storage.omega);
    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = shambase::get_check_ref(storage.pressure).get_field(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs
            = shambase::get_check_ref(storage.soundspeed).get_field(cur_p.id_patch).get_buf();

        sham::DeviceBuffer<Tvec> &buf_dB_on_rho   = pdat.get_field_buf_ref<Tvec>(idB_on_rho);
        sham::DeviceBuffer<Tscal> &buf_dpsi_on_ch = pdat.get_field_buf_ref<Tscal>(idpsi_on_ch);
        sham::DeviceBuffer<Tscal> &buf_drho_dt    = pdat.get_field_buf_ref<Tscal>(idrho_dt);
        // logger::raw_ln("charged dB dpsi");

        sham::DeviceBuffer<Tvec> &buf_B_on_rho = mpdat.get_field_buf_ref<Tvec>(iB_on_rho_interf);
        sham::DeviceBuffer<Tscal> &buf_psi_on_ch
            = mpdat.get_field_buf_ref<Tscal>(ipsi_on_ch_interf);

        // logger::raw_ln("charged B psi");
        //  ADD curlBBBBBBBBB

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz        = buf_xyz.get_read_access(depends_list);
        auto axyz       = buf_axyz.get_write_access(depends_list);
        auto du         = buf_duint.get_write_access(depends_list);
        auto vxyz       = buf_vxyz.get_read_access(depends_list);
        auto hpart      = buf_hpart.get_read_access(depends_list);
        auto omega      = buf_omega.get_read_access(depends_list);
        auto u          = buf_uint.get_read_access(depends_list);
        auto pressure   = buf_pressure.get_read_access(depends_list);
        auto cs         = buf_cs.get_read_access(depends_list);
        auto B_on_rho   = buf_B_on_rho.get_read_access(depends_list);
        auto psi_on_ch  = buf_psi_on_ch.get_read_access(depends_list);
        auto dB_on_rho  = buf_dB_on_rho.get_write_access(depends_list);
        auto dpsi_on_ch = buf_dpsi_on_ch.get_write_access(depends_list);
        auto drho_dt    = buf_drho_dt.get_write_access(depends_list);

        Tvec *mag_pressure
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tvec>(imag_pressure).get_write_access(depends_list)
                  : nullptr;
        Tvec *mag_tension
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tvec>(imag_tension).get_write_access(depends_list)
                  : nullptr;
        Tvec *gas_pressure
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tvec>(igas_pressure).get_write_access(depends_list)
                  : nullptr;
        Tvec *tensile_corr
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tvec>(itensile_corr).get_write_access(depends_list)
                  : nullptr;

        Tscal *psi_propag
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tscal>(ipsi_propag).get_write_access(depends_list)
                  : nullptr;
        Tscal *psi_diff
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tscal>(ipsi_diff).get_write_access(depends_list)
                  : nullptr;
        Tscal *psi_cons
            = (do_MHD_debug)
                  ? pdat.get_field_buf_ref<Tscal>(ipsi_cons).get_write_access(depends_list)
                  : nullptr;

        Tscal *u_mhd = (do_MHD_debug)
                           ? pdat.get_field_buf_ref<Tscal>(iu_mhd).get_write_access(depends_list)
                           : nullptr;

        auto ploop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass     = solver_config.gpart_mass;
            const Tscal sigma_mhd = cfg.sigma_mhd;
            const Tscal alpha_u   = cfg.alpha_u;

            shamlog_debug_ln("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "");
            shamlog_debug_sycl_ln("deriv kernel", "sigma_mhd  :", sigma_mhd);
            shamlog_debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            shamlog_debug_ln("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "");

            tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute MHD", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a       = hpart[id_a];
                Tvec xyz_a      = xyz[id_a];
                Tvec vxyz_a     = vxyz[id_a];
                Tscal P_a       = pressure[id_a];
                Tscal cs_a      = cs[id_a];
                Tscal omega_a   = omega[id_a];
                const Tscal u_a = u[id_a];

                Tscal rho_a     = rho_h(pmass, h_a, Kernel::hfactd);
                Tscal rho_a_sq  = rho_a * rho_a;
                Tscal rho_a_inv = 1. / rho_a;

                Tvec B_a         = B_on_rho[id_a] * rho_a;
                Tscal v_alfven_a = sycl::sqrt(sycl::dot(B_a, B_a) / (mu_0 * rho_a));
                Tscal v_shock_a  = sycl::sqrt(cs_a * cs_a + v_alfven_a * v_alfven_a);
                Tscal psi_a      = psi_on_ch[id_a] * v_shock_a;

                Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

                Tvec force_pressure{0, 0, 0};
                Tscal tmpdU_pressure = 0;
                Tvec magnetic_eq{0, 0, 0};
                Tscal psi_eq  = 0;
                Tscal drho_eq = 0;

                Tvec mag_pressure_term{0, 0, 0};
                Tvec mag_tension_term{0, 0, 0};
                Tvec gas_pressure_term{0, 0, 0};
                Tvec tensile_corr_term{0, 0, 0};

                Tscal psi_propag_term = 0;
                Tscal psi_diff_term   = 0;
                Tscal psi_cons_term   = 0;

                Tscal u_mhd_term = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // compute only omega_a
                    Tvec dr    = xyz_a - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    Tvec vxyz_b     = vxyz[id_b];
                    const Tscal u_b = u[id_b];
                    Tscal P_b       = pressure[id_b];
                    Tscal omega_b   = omega[id_b];
                    Tscal cs_b      = cs[id_b];

                    Tscal rab = sycl::sqrt(rab2);

                    Tscal rho_b      = rho_h(pmass, h_b, Kernel::hfactd);
                    Tvec B_b         = B_on_rho[id_b] * rho_b;
                    Tscal v_alfven_b = sycl::sqrt(sycl::dot(B_b, B_b) / (mu_0 * rho_b));
                    Tscal v_shock_b  = sycl::sqrt(cs_b * cs_b + v_alfven_b * v_alfven_b);
                    Tscal psi_b      = psi_on_ch[id_b] * v_shock_b;
                    // const Tscal alpha_a = alpha_AV;
                    // const Tscal alpha_b = alpha_AV;
                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // Tscal sigma_mhd = 0.3;
                    shamrock::sph::mhd::add_to_derivs_spmhd<Kernel, Tvec, Tscal>(
                        pmass,
                        dr,
                        rab,
                        rho_a,
                        rho_a_sq,
                        omega_a_rho_a_inv,
                        rho_a_inv,
                        rho_b,
                        omega_a,
                        omega_b,
                        Fab_a,
                        Fab_b,
                        vxyz_a,
                        vxyz_b,
                        u_a,
                        u_b,
                        P_a,
                        P_b,
                        cs_a,
                        cs_b,
                        h_a,
                        h_b,

                        alpha_u,

                        B_a,
                        B_b,

                        psi_a,
                        psi_b,

                        mu_0,
                        sigma_mhd,

                        force_pressure,
                        tmpdU_pressure,
                        magnetic_eq,
                        psi_eq,
                        drho_eq,
                        mag_pressure_term,
                        mag_tension_term,
                        gas_pressure_term,
                        tensile_corr_term,

                        psi_propag_term,
                        psi_diff_term,
                        psi_cons_term,
                        u_mhd_term);
                });

                axyz[id_a]       = force_pressure;
                du[id_a]         = tmpdU_pressure;
                dB_on_rho[id_a]  = magnetic_eq;
                dpsi_on_ch[id_a] = psi_eq - psi_a / h_a;
                drho_dt[id_a]    = drho_eq;

                if (do_MHD_debug) {
                    mag_pressure[id_a] = mag_pressure_term;
                    mag_tension[id_a]  = mag_tension_term;
                    gas_pressure[id_a] = gas_pressure_term;
                    tensile_corr[id_a] = tensile_corr_term;

                    psi_propag[id_a] = psi_propag_term;
                    psi_diff[id_a]   = psi_diff_term;
                    psi_cons[id_a]   = -psi_a / h_a;

                    u_mhd[id_a] = u_mhd_term;
                }
            });
        });

        buf_xyz.complete_event_state(e);
        buf_axyz.complete_event_state(e);
        buf_duint.complete_event_state(e);
        buf_vxyz.complete_event_state(e);
        buf_hpart.complete_event_state(e);
        buf_omega.complete_event_state(e);
        buf_uint.complete_event_state(e);
        buf_pressure.complete_event_state(e);
        buf_cs.complete_event_state(e);
        buf_B_on_rho.complete_event_state(e);
        buf_psi_on_ch.complete_event_state(e);
        buf_dB_on_rho.complete_event_state(e);
        buf_dpsi_on_ch.complete_event_state(e);
        buf_drho_dt.complete_event_state(e);

        if (do_MHD_debug) {
            pdat.get_field_buf_ref<Tvec>(imag_pressure).complete_event_state(e);
            pdat.get_field_buf_ref<Tvec>(imag_tension).complete_event_state(e);
            pdat.get_field_buf_ref<Tvec>(igas_pressure).complete_event_state(e);
            pdat.get_field_buf_ref<Tvec>(itensile_corr).complete_event_state(e);

            pdat.get_field_buf_ref<Tscal>(ipsi_propag).complete_event_state(e);
            pdat.get_field_buf_ref<Tscal>(ipsi_diff).complete_event_state(e);
            pdat.get_field_buf_ref<Tscal>(ipsi_cons).complete_event_state(e);

            pdat.get_field_buf_ref<Tscal>(iu_mhd).complete_event_state(e);
        }

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);
    });
}

using namespace shammath;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M4>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M6>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M8>;

template class shammodels::sph::modules::UpdateDerivs<f64_3, C2>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, C4>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, C6>;
