// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperator.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shammodels/sph/modules/DiffOperator.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_divv() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;
    
    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idivv = pdl.get_field_idx<Tscal>("divv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_divv  = pdat.get_field_buf_ref<Tscal>(idivv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute divv"};
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(pcache, cgh);

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
                sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                sycl::accessor omega{buf_omega, cgh, sycl::read_only};
                sycl::accessor divv{buf_divv, cgh, sycl::write_only, sycl::no_init};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(),"compute divv", [=](i32 id_a){

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
        }
    });
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperators<Tvec, SPHKernel>::update_curlv() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch = mpdat.get(cur_p.id_patch);
        PatchData &mpdat              = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_vxyz   = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tscal> &buf_hpart = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint  = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tvec> &buf_curlv  = pdat.get_field_buf_ref<Tvec>(icurlv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute curlv"};
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(pcache, cgh);

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
                sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                sycl::accessor omega{buf_omega, cgh, sycl::read_only};
                sycl::accessor curlv{buf_curlv, cgh, sycl::write_only, sycl::no_init};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(),"compute curlv", [=](i32 id_a){

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
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperators<f64_3, M4>;
template class shammodels::sph::modules::DiffOperators<f64_3, M6>;