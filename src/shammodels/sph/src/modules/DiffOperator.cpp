// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_divv() {

    StackEntry stack_loc{};
    shamlog_debug_ln("SPH", "Updating divv");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idivv = pdl.get_field_idx<Tscal>("divv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tscal> &buf_divv  = pdat.get_field_buf_ref<Tscal>(idivv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute divv"};

            sham::EventList depends_list;

            auto xyz        = buf_xyz.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto omega      = buf_omega.get_read_access(depends_list);
            auto divv       = buf_divv.get_write_access(depends_list);
            auto ploop_ptrs = pcache.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute divv", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec vxyz_a    = vxyz[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tscal sum_nabla_v = 0;

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec vxyz_b = vxyz[id_b];
                        Tvec v_ab   = vxyz_a - vxyz_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        sum_nabla_v += pmass * sycl::dot(v_ab, dWab_a);
                    });

                    divv[id_a] = -inv_rho_omega_a * sum_nabla_v;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_divv.complete_event_state(e);

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        }
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_curlv() {

    StackEntry stack_loc{};
    shamlog_debug_ln("SPH", "Updating curlv");

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl();

    shambase::DistributedData<PatchDataLayer> &mpdats = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ihpart_interf = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf  = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf  = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataLayer &mpdat = mpdats.get(cur_p.id_patch);

        sham::DeviceBuffer<Tvec> &buf_xyz
            = merged_xyzh.get(cur_p.id_patch).template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sham::DeviceBuffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sham::DeviceBuffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sham::DeviceBuffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sham::DeviceBuffer<Tvec> &buf_curlv  = pdat.get_field_buf_ref<Tvec>(icurlv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute curlv"};

            sham::EventList depends_list;
            auto xyz        = buf_xyz.get_read_access(depends_list);
            auto vxyz       = buf_vxyz.get_read_access(depends_list);
            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto omega      = buf_omega.get_read_access(depends_list);
            auto curlv      = buf_curlv.get_write_access(depends_list);
            auto ploop_ptrs = pcache.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parallel_for(cgh, pdat.get_obj_cnt(), "compute curlv", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
                    Tscal sum_du_a = 0;
                    Tscal h_a      = hpart[id_a];
                    Tvec xyz_a     = xyz[id_a];
                    Tvec vxyz_a    = vxyz[id_a];
                    Tscal omega_a  = omega[id_a];

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);
                    // Tscal rho_a_sq  = rho_a * rho_a;
                    // Tscal rho_a_inv = 1. / rho_a;
                    Tscal inv_rho_omega_a = 1. / (omega_a * rho_a);

                    Tvec sum_nabla_cross_v{};

                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr    = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab   = sycl::sqrt(rab2);
                        Tvec vxyz_b = vxyz[id_b];
                        Tvec v_ab   = vxyz_a - vxyz_b;

                        Tvec r_ab_unit = dr / rab;

                        if (rab < 1e-9) {
                            r_ab_unit = {0, 0, 0};
                        }

                        Tvec dWab_a = Kernel::dW_3d(rab, h_a) * r_ab_unit;

                        sum_nabla_cross_v += pmass * sycl::cross(v_ab, dWab_a);
                    });

                    curlv[id_a] = -inv_rho_omega_a * sum_nabla_cross_v;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_vxyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            buf_omega.complete_event_state(e);
            buf_curlv.complete_event_state(e);

            sham::EventList resulting_events;
            resulting_events.add_event(e);
            pcache.complete_event_state(resulting_events);
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperators<f64_3, M4>;
template class shammodels::sph::modules::DiffOperators<f64_3, M6>;
template class shammodels::sph::modules::DiffOperators<f64_3, M8>;

template class shammodels::sph::modules::DiffOperators<f64_3, C2>;
template class shammodels::sph::modules::DiffOperators<f64_3, C4>;
template class shammodels::sph::modules::DiffOperators<f64_3, C6>;
