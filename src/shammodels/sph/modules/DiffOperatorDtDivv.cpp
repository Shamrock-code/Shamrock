// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperatorDtDivv.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/DiffOperatorDtDivv.hpp"
#include "shambase/memory.hpp"
#include "shammath/matrix.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::DiffOperatorDtDivv<Tvec, SPHKernel>::update_dtdivv() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 iaxyz      = pdl.get_field_idx<Tvec>("axyz");

    sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();

    using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

    shambase::Timer time_interf;
    time_interf.start();

    // exchange the actual state of the acceleration
    // this should be ran only after acceleration update
    // ideally i wouldn't want to do this but due to the formulation
    // of the Culhen & Dehnen switch we need the divergence of the acceleration
    // the following calls exchange the acceleration and make a merged field out of it
    auto a_interf = ghost_handle.template build_interface_native<PatchDataField<Tvec>>(
        storage.ghost_patch_cache.get(),
        [&](u64 sender,
            u64 /*receiver*/,
            InterfaceBuildInfos binfo,
            sycl::buffer<u32> &buf_idx,
            u32 cnt) -> PatchDataField<Tvec> {
            PatchData &sender_patch = scheduler().patch_data.get_pdat(sender);

            PatchDataField<Tvec> &sender_axyz = sender_patch.get_field<Tvec>(iaxyz);

            return sender_axyz.make_new_from_subset(buf_idx, cnt);
        });

    shambase::DistributedDataShared<PatchDataField<Tvec>> interf_pdat =
        ghost_handle.communicate_pdatfield(std::move(a_interf), 1);

    shambase::DistributedData<PatchDataField<Tvec>> merged_a =
        ghost_handle.template merge_native<PatchDataField<Tvec>, PatchDataField<Tvec>>(
            std::move(interf_pdat),
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
                PatchDataField<Tvec> &axyz = pdat.get_field<Tvec>(iaxyz);

                return axyz.duplicate();
            },
            [](PatchDataField<Tvec> &mpdat, PatchDataField<Tvec> &pdat_interf) {
                mpdat.insert(pdat_interf);
            });

    time_interf.end();
    storage.timings_details.interface += time_interf.elasped_sec();

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    auto &merged_xyzh = storage.merged_xyzh.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");
    u32 ivxyz_interf                               = ghost_layout.get_field_idx<Tvec>("vxyz");
    u32 iomega_interf                              = ghost_layout.get_field_idx<Tscal>("omega");

    const u32 idtdivv = pdl.get_field_idx<Tscal>("dtdivv");
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        MergedPatchData &merged_patch     = mpdat.get(cur_p.id_patch);
        PatchDataField<Tvec> &merged_axyz = merged_a.get(cur_p.id_patch);
        PatchData &mpdat                  = merged_patch.pdat;

        sycl::buffer<Tvec> &buf_xyz =
            shambase::get_check_ref(merged_xyzh.get(cur_p.id_patch).field_pos.get_buf());
        sycl::buffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz_interf);
        sycl::buffer<Tvec> &buf_axyz = shambase::get_check_ref(merged_axyz.get_buf());

        sycl::buffer<Tscal> &buf_hpart  = mpdat.get_field_buf_ref<Tscal>(ihpart_interf);
        sycl::buffer<Tscal> &buf_omega  = mpdat.get_field_buf_ref<Tscal>(iomega_interf);
        sycl::buffer<Tscal> &buf_uint   = mpdat.get_field_buf_ref<Tscal>(iuint_interf);
        sycl::buffer<Tscal> &buf_dtdivv = pdat.get_field_buf_ref<Tscal>(idtdivv);

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(cur_p.id_patch);

        /////////////////////////////////////////////

        {
            NamedStackEntry tmppp{"compute dtdivv"};
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                const Tscal pmass = gpart_mass;

                tree::ObjectCacheIterator particle_looper(pcache, cgh);

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor vxyz{buf_vxyz, cgh, sycl::read_only};
                sycl::accessor axyz{buf_axyz, cgh, sycl::read_only};
                sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};
                sycl::accessor omega{buf_omega, cgh, sycl::read_only};
                sycl::accessor dtdivv{buf_dtdivv, cgh, sycl::write_only, sycl::no_init};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute dtdivv", [=](i32 id_a) {
                    using namespace shamrock::sph;

                    Tvec sum_axyz  = {0, 0, 0};
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

                    std::array<Tvec, dim> Rij_a{Tvec{0}, Tvec{0}, Tvec{0}};

                    std::array<Tvec, dim> Rij_a_dvk_dxj{Tvec{0}, Tvec{0}, Tvec{0}};

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
                        Rij_a[0] += r_ab.x() * mdWab_b;
                        Rij_a[1] += r_ab.y() * mdWab_b;
                        Rij_a[2] += r_ab.z() * mdWab_b;

                        Rij_a_dvk_dxj[0] += v_ab * mdWab_b.x();
                        Rij_a_dvk_dxj[1] += v_ab * mdWab_b.y();
                        Rij_a_dvk_dxj[2] += v_ab * mdWab_b.z();

                        sum_nabla_a += sycl::dot(a_ab, mdWab_b);
                    });

                    std::array<Tvec, 3> invRij = shammath::compute_inv_33(Rij_a);

                    std::array<Tvec, 3> dvi_dxk = shammath::mat_prod_33(invRij, Rij_a_dvk_dxj);

                    Tscal tens_nablav =
                        dvi_dxk[0].x() * dvi_dxk[0].x() + dvi_dxk[1].x() * dvi_dxk[0].y() +
                        dvi_dxk[2].x() * dvi_dxk[0].z() + dvi_dxk[0].y() * dvi_dxk[1].x() +
                        dvi_dxk[1].y() * dvi_dxk[1].y() + dvi_dxk[2].y() * dvi_dxk[1].z() +
                        dvi_dxk[0].z() * dvi_dxk[2].x() + dvi_dxk[1].z() * dvi_dxk[2].y() +
                        dvi_dxk[2].z() * dvi_dxk[2].z();

                    dtdivv[id_a] = (-inv_rho_omega_a * sum_nabla_a) - tens_nablav;
                });
            });
        }
    });
}

using namespace shammath;
template class shammodels::sph::modules::DiffOperatorDtDivv<f64_3, M4>;
template class shammodels::sph::modules::DiffOperatorDtDivv<f64_3, M6>;