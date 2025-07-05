// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperatorDtDivv.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shammath/matrix_legacy.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/DiffOperatorDtDivv.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

#define ZVEC shambase::VectorProperties<Tvec>::get_zero()

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperatorDtDivv<Tvec, SPHKernel>::update_dtdivv(
    bool also_do_div_curl_v) {

    StackEntry stack_loc{};

    shamlog_debug_ln("SPH", "Updating dt divv");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iaxyz_interf                               = ghost_layout.get_field_idx<Tvec>("axyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idtdivv = pdl.get_field_idx<Tscal>("dtdivv");

    const u32 idivv  = pdl.get_field_idx<Tscal>("divv");
    const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sham::DeviceBuffer<Tvec> &buf_xyz  = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tvec> &buf_axyz = mpdat.get_field_buf_ref<Tvec>(iaxyz_interf);

        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);

        sham::DeviceBuffer<Tscal> &buf_divv   = pdat.get_field_buf_ref<Tscal>(idivv);
        sham::DeviceBuffer<Tvec> &buf_curlv   = pdat.get_field_buf_ref<Tvec>(icurlv);
        sham::DeviceBuffer<Tscal> &buf_dtdivv = pdat.get_field_buf_ref<Tscal>(idtdivv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        if (!also_do_div_curl_v) {
            NamedStackEntry tmppp{"compute dtdivv"};

            sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();

            sham::EventList depends_list;

            auto xyz        = buf_xyz.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto axyz       = buf_axyz.get_read_access(depends_list);
            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto omega      = buf_omega.get_read_access(depends_list);
            auto dtdivv     = buf_dtdivv.get_write_access(depends_list);
            auto ploop_ptrs = pcache.get_read_access(depends_list);

            auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute dtdivv", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = ZVEC;
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec vxyz_a    = vxyz[id_a];
                    Tvec axyz_a    = axyz[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tscal sum_nabla_a = 0;

                    std::array<Tvec, dim> Rij_a{ZVEC, ZVEC, ZVEC};

                    std::array<Tvec, dim> Rij_a_dvk_dxj{ZVEC, ZVEC, ZVEC};
                    std::array<Tvec, dim> Rij_a_dak_dxj{ZVEC, ZVEC, ZVEC};

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec r_ab  = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(r_ab, r_ab);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec vxyz_b = vxyz[id_b];
                        Tvec axyz_b = axyz[id_b];
                        Tvec v_ab   = vxyz_a - vxyz_b;
                        Tvec a_ab   = axyz_a - axyz_b;

                        Tvec r_ab_unit = r_ab / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        Tvec mdWab_b = dWab_a * pmass;

                        static_assert(dim == 3, "this is only implemented for dim 3");
                        Rij_a[0] -= r_ab.x() * mdWab_b;
                        Rij_a[1] -= r_ab.y() * mdWab_b;
                        Rij_a[2] -= r_ab.z() * mdWab_b;

                        Rij_a_dvk_dxj[0] -= v_ab * mdWab_b.x();
                        Rij_a_dvk_dxj[1] -= v_ab * mdWab_b.y();
                        Rij_a_dvk_dxj[2] -= v_ab * mdWab_b.z();

                        Rij_a_dak_dxj[0] -= a_ab * mdWab_b.x();
                        Rij_a_dak_dxj[1] -= a_ab * mdWab_b.y();
                        Rij_a_dak_dxj[2] -= a_ab * mdWab_b.z();

                        // sum_nabla_a += sycl::dot(a_ab, mdWab_b);
                    });

                    std::array<Tvec, 3> invRij = shammath::compute_inv_33(Rij_a);

                    std::array<Tvec, 3> dvi_dxk = shammath::mat_prod_33(invRij, Rij_a_dvk_dxj);
                    std::array<Tvec, 3> dai_dxk = shammath::mat_prod_33(invRij, Rij_a_dak_dxj);

                    Tscal div_ai = dai_dxk[0].x() + dai_dxk[1].y() + dai_dxk[2].z();
                    Tscal div_vi = dvi_dxk[0].x() + dvi_dxk[1].y() + dvi_dxk[2].z();
                    Tvec curl_vi
                        = {dvi_dxk[1].z() - dvi_dxk[2].y(),
                           dvi_dxk[2].x() - dvi_dxk[0].z(),
                           dvi_dxk[0].y() - dvi_dxk[1].x()};

                    Tscal tens_nablav
                        = dvi_dxk[0].x() * dvi_dxk[0].x() + dvi_dxk[1].x() * dvi_dxk[0].y()
                          + dvi_dxk[2].x() * dvi_dxk[0].z() + dvi_dxk[0].y() * dvi_dxk[1].x()
                          + dvi_dxk[1].y() * dvi_dxk[1].y() + dvi_dxk[2].y() * dvi_dxk[1].z()
                          + dvi_dxk[0].z() * dvi_dxk[2].x() + dvi_dxk[1].z() * dvi_dxk[2].y()
                          + dvi_dxk[2].z() * dvi_dxk[2].z();

                    // divv[id_a] = div_vi;
                    // curlv[id_a] = curl_vi;
                    dtdivv[id_a] = div_ai - tens_nablav;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_axyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_dtdivv.complete_event_state(e);
            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);

        } else {

            NamedStackEntry tmppp{"compute dtdivv + divcurl v"};

            sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();

            sham::EventList depends_list;

            auto xyz        = buf_xyz.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto axyz       = buf_axyz.get_read_access(depends_list);
            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto omega      = buf_omega.get_read_access(depends_list);
            auto divv       = buf_divv.get_write_access(depends_list);
            auto curlv      = buf_curlv.get_write_access(depends_list);
            auto dtdivv     = buf_dtdivv.get_write_access(depends_list);
            auto ploop_ptrs = pcache.get_read_access(depends_list);

            auto e = queue.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(
                    cgh, pdat.get_obj_cnt(), "compute dtdivv + divcurl v", [=](i32 id_a) {
                        using namespace shamrock::sph;

                        Tvec sum_axyz  = ZVEC;
                        Tscal sum_du_a = 0;
                        Tscal h_a      = hpart[id_a];
                        Tvec xyz_a     = xyz[id_a];
                        Tvec vxyz_a    = vxyz[id_a];
                        Tvec axyz_a    = axyz[id_a];
                        Tscal omega_a  = omega[id_a];

                        Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                        // Tscal rho_a_sq  = rho_a * rho_a;
                        // Tscal rho_a_inv = 1. / rho_a;
                        Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                        Tscal sum_nabla_a = 0;

                        std::array<Tvec, dim> Rij_a{ZVEC, ZVEC, ZVEC};

                        std::array<Tvec, dim> Rij_a_dvk_dxj{ZVEC, ZVEC, ZVEC};
                        std::array<Tvec, dim> Rij_a_dak_dxj{ZVEC, ZVEC, ZVEC};

                        Tscal sum_nabla_v = 0;
                        Tvec sum_nabla_cross_v{};

                        particle_looper.for_each_object(id_a, [&](u32 id_b) {
                            // compute only omega_a
                            Tvec r_ab  = xyz_a - xyz[id_b];
                            Tscal rab2 = sycl::dot(r_ab, r_ab);
                            Tscal h_b  = hpart[id_b];

                            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                                return;
                            }

                            Tscal rab   = sycl::sqrt(rab2);
                            Tvec vxyz_b = vxyz[id_b];
                            Tvec axyz_b = axyz[id_b];
                            Tvec v_ab   = vxyz_a - vxyz_b;
                            Tvec a_ab   = axyz_a - axyz_b;

                            Tvec r_ab_unit = r_ab / rab;

                            if (rab < 1e-9) {
                                r_ab_unit = ZVEC;
                            }

                            Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                            Tvec mdWab_b = dWab_a * pmass;

                            static_assert(dim == 3, "this is only implemented for dim 3");
                            Rij_a[0] -= r_ab.x() * mdWab_b;
                            Rij_a[1] -= r_ab.y() * mdWab_b;
                            Rij_a[2] -= r_ab.z() * mdWab_b;

                            Rij_a_dvk_dxj[0] -= v_ab * mdWab_b.x();
                            Rij_a_dvk_dxj[1] -= v_ab * mdWab_b.y();
                            Rij_a_dvk_dxj[2] -= v_ab * mdWab_b.z();

                            Rij_a_dak_dxj[0] -= a_ab * mdWab_b.x();
                            Rij_a_dak_dxj[1] -= a_ab * mdWab_b.y();
                            Rij_a_dak_dxj[2] -= a_ab * mdWab_b.z();

                            // sum_nabla_a += sycl::dot(a_ab, mdWab_b);
                            sum_nabla_v += pmass * sycl::dot(v_ab, dWab_a);
                            sum_nabla_cross_v += pmass * sycl::cross(v_ab, dWab_a);
                        });

                        std::array<Tvec, 3> invRij = shammath::compute_inv_33(Rij_a);

                        std::array<Tvec, 3> dvi_dxk = shammath::mat_prod_33(invRij, Rij_a_dvk_dxj);
                        std::array<Tvec, 3> dai_dxk = shammath::mat_prod_33(invRij, Rij_a_dak_dxj);

                        Tscal div_ai = dai_dxk[0].x() + dai_dxk[1].y() + dai_dxk[2].z();
                        Tscal div_vi = dvi_dxk[0].x() + dvi_dxk[1].y() + dvi_dxk[2].z();
                        Tvec curl_vi
                            = {dvi_dxk[1].z() - dvi_dxk[2].y(),
                               dvi_dxk[2].x() - dvi_dxk[0].z(),
                               dvi_dxk[0].y() - dvi_dxk[1].x()};

                        Tscal tens_nablav
                            = dvi_dxk[0].x() * dvi_dxk[0].x() + dvi_dxk[1].x() * dvi_dxk[0].y()
                              + dvi_dxk[2].x() * dvi_dxk[0].z() + dvi_dxk[0].y() * dvi_dxk[1].x()
                              + dvi_dxk[1].y() * dvi_dxk[1].y() + dvi_dxk[2].y() * dvi_dxk[1].z()
                              + dvi_dxk[0].z() * dvi_dxk[2].x() + dvi_dxk[1].z() * dvi_dxk[2].y()
                              + dvi_dxk[2].z() * dvi_dxk[2].z();

                        // divv[id_a] = div_vi;
                        // curlv[id_a] = curl_vi;
                        divv[id_a]   = -inv_rho_omega_a * sum_nabla_v;
                        curlv[id_a]  = -inv_rho_omega_a * sum_nabla_cross_v;
                        dtdivv[id_a] = div_ai - tens_nablav;
                    });
            });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_axyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_divv.complete_event_state(e);
            buf_curlv.complete_event_state(e);
            buf_dtdivv.complete_event_state(e);
            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperatorDtDivv<f64_3, M4>;
template class shammodels::sph::modules::DiffOperatorDtDivv<f64_3, M6>;
template class shammodels::sph::modules::DiffOperatorDtDivv<f64_3, M8>;
