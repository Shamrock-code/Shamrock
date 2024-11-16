// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file UpdateDerivs.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/UpdateDerivs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs() {

    Cfg_AV cfg_av = solver_config.artif_viscosity;

    if (None *v = std::get_if<None>(&cfg_av.config)) {
        shambase::throw_unimplemented();
    } else if (Constant *v = std::get_if<Constant>(&cfg_av.config)) {
        update_derivs_constantAV(*v);
    } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&cfg_av.config)) {
        update_derivs_mm97(*v);
    } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&cfg_av.config)) {
        update_derivs_cd10(*v);
    } else if (ConstantDisc *v = std::get_if<ConstantDisc>(&cfg_av.config)) {
        update_derivs_disc_visco(*v);
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

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sham::DeviceBuffer<Tvec> &buf_xyz    = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(cur_p.id_patch);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz      = buf_xyz.get_read_access(depends_list);
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto du       = buf_duint.get_write_access(depends_list);
        auto vxyz     = buf_vxyz.get_read_access(depends_list);
        auto hpart    = buf_hpart.get_read_access(depends_list);
        auto omega    = buf_omega.get_read_access(depends_list);
        auto u        = buf_uint.get_read_access(depends_list); // TODO rename to uint
        auto pressure = buf_pressure.get_read_access(depends_list);
        auto cs       = buf_cs.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal alpha_u  = cfg.alpha_u;
            const Tscal alpha_AV = cfg.alpha_AV;
            const Tscal beta_AV  = cfg.beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "alpha_AV :", alpha_AV);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force cte AV", [=](u64 gid) {
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
                    Tvec v_ab       = vxyz_a - vxyz_b;
                    const Tscal u_b = u[id_b];

                    Tvec r_ab_unit = dr / rab;

                    if (rab < 1e-9) {
                        r_ab_unit = {0, 0, 0};
                    }

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);
                    Tscal P_b   = pressure[id_b];
                    // f32 P_b     = cs * cs * rho_b;
                    Tscal omega_b = omega[id_b];
                    Tscal cs_b    = cs[id_b];

                    /////////////////
                    // internal energy update
                    //  scalar : f32  | vector : f32_3
                    const Tscal alpha_a = alpha_AV;
                    const Tscal alpha_b = alpha_AV;

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        h_a, h_b,
                        beta_AV, alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
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
    });
}
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_mm97(VaryingMM97 cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz      = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint     = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint    = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart    = pdl.get_field_idx<Tscal>("hpart");
    const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");
    u32 ialpha_AV_interf                           = ghost_layout.get_field_idx<Tscal>("alpha_AV");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sham::DeviceBuffer<Tvec> &buf_xyz    = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_alpha_AV
            = storage.alpha_av_ghost.get().get(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(cur_p.id_patch);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz      = buf_xyz.get_read_access(depends_list);
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto du       = buf_duint.get_write_access(depends_list);
        auto vxyz     = buf_vxyz.get_read_access(depends_list);
        auto hpart    = buf_hpart.get_read_access(depends_list);
        auto omega    = buf_omega.get_read_access(depends_list);
        auto u        = buf_uint.get_read_access(depends_list);
        auto pressure = buf_pressure.get_read_access(depends_list);
        auto alpha_AV = buf_alpha_AV.get_read_access(depends_list);
        auto cs       = buf_cs.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass   = solver_config.gpart_mass;
            const Tscal alpha_u = cfg.alpha_u;
            const Tscal beta_AV = cfg.beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force MM97 AV", [=](u64 gid) {
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
                    Tvec v_ab       = vxyz_a - vxyz_b;
                    const Tscal u_b = u[id_b];

                    Tvec r_ab_unit = dr / rab;

                    if (rab < 1e-9) {
                        r_ab_unit = {0, 0, 0};
                    }

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

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                       h_a, h_b,
                        beta_AV, alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
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
    });
}
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_cd10(VaryingCD10 cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch        = mpdat.get(cur_p.id_patch);
        PatchData &mpdat                     = merged_patch.pdat;
        sham::DeviceBuffer<Tvec> &buf_xyz    = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_alpha_AV
            = storage.alpha_av_ghost.get().get(cur_p.id_patch).get_buf();
        sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(cur_p.id_patch);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz      = buf_xyz.get_read_access(depends_list);
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto du       = buf_duint.get_write_access(depends_list);
        auto vxyz     = buf_vxyz.get_read_access(depends_list);
        auto hpart    = buf_hpart.get_read_access(depends_list);
        auto omega    = buf_omega.get_read_access(depends_list);
        auto u        = buf_uint.get_read_access(depends_list);
        auto pressure = buf_pressure.get_read_access(depends_list);
        auto alpha_AV = buf_alpha_AV.get_read_access(depends_list);
        auto cs       = buf_cs.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass   = solver_config.gpart_mass;
            const Tscal alpha_u = cfg.alpha_u;
            const Tscal beta_AV = cfg.beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force CD10 AV", [=](u64 gid) {
                u32 id_a = (u32) gid;

                using namespace shamrock::sph;

                Tvec sum_axyz  = {0, 0, 0};
                Tscal sum_du_a = 0;

                Tscal h_a           = hpart[id_a];
                Tvec xyz_a          = xyz[id_a];
                Tvec vxyz_a         = vxyz[id_a];
                Tscal P_a           = pressure[id_a];
                Tscal cs_a          = cs[id_a];
                Tscal omega_a       = omega[id_a];
                const Tscal u_a     = u[id_a];
                const Tscal alpha_a = alpha_AV[id_a];

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

                    Tvec vxyz_b         = vxyz[id_b];
                    const Tscal u_b     = u[id_b];
                    Tscal P_b           = pressure[id_b];
                    Tscal omega_b       = omega[id_b];
                    const Tscal alpha_b = alpha_AV[id_b];
                    Tscal cs_b          = cs[id_b];

                    Tscal rab = sycl::sqrt(rab2);

                    Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

                    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                    Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        h_a, h_b,

                        beta_AV, alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
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
        buf_alpha_AV.complete_event_state(e);
        buf_cs.complete_event_state(e);
    });
}
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDerivs<Tvec, SPHKernel>::update_derivs_disc_visco(
    ConstantDisc cfg) {
    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 ixyz   = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz  = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz  = pdl.get_field_idx<Tvec>("axyz");
    const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
    const u32 iduint = pdl.get_field_idx<Tscal>("duint");
    const u32 ihpart = pdl.get_field_idx<Tscal>("hpart");

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    auto &merged_xyzh                                 = storage.merged_xyzh.get();
    ComputeField<Tscal> &omega                        = storage.omega.get();
    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sham::DeviceBuffer<Tvec> &buf_xyz    = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_axyz   = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tscal> &buf_duint = pdat.get_field_buf_ref<Tscal>(iduint);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_pressure
            = storage.pressure.get().get_buf_check(cur_p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(cur_p.id_patch);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto xyz      = buf_xyz.get_read_access(depends_list);
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto du       = buf_duint.get_write_access(depends_list);
        auto vxyz     = buf_vxyz.get_read_access(depends_list);
        auto hpart    = buf_hpart.get_read_access(depends_list);
        auto omega    = buf_omega.get_read_access(depends_list);
        auto u        = buf_uint.get_read_access(depends_list);
        auto pressure = buf_pressure.get_read_access(depends_list);
        auto cs       = buf_cs.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const Tscal pmass    = solver_config.gpart_mass;
            const Tscal alpha_AV = cfg.alpha_AV;
            const Tscal alpha_u  = cfg.alpha_u;
            const Tscal beta_AV  = cfg.beta_AV;

            logger::debug_sycl_ln("deriv kernel", "alpha_AV  :", alpha_AV);
            logger::debug_sycl_ln("deriv kernel", "alpha_u  :", alpha_u);
            logger::debug_sycl_ln("deriv kernel", "beta_AV  :", beta_AV);

            // tree::ObjectIterator particle_looper(tree,cgh);

            // tree::LeafCacheObjectIterator
            // particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            tree::ObjectCacheIterator particle_looper(pcache, cgh);

            // sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute force disc", [=](u64 gid) {
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

                    // clang-format off
                    add_to_derivs_sph_artif_visco_cond<Kernel,Tvec, Tscal, shamrock::sph::Disc>(
                        pmass,
                        dr, rab,
                        rho_a, rho_a_sq, omega_a_rho_a_inv, rho_a_inv, rho_b,
                        omega_a, omega_b,
                        Fab_a, Fab_b,
                        vxyz_a, vxyz_b,
                        u_a, u_b,
                        P_a, P_b,
                        cs_a, cs_b,
                        alpha_a, alpha_b,
                        h_a, h_b,

                        beta_AV, alpha_u,

                        force_pressure,
                        tmpdU_pressure);
                    // clang-format on
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
    });
}

using namespace shammath;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M4>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M6>;
template class shammodels::sph::modules::UpdateDerivs<f64_3, M8>;
