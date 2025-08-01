// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExternalForces.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamunits/Constants.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::compute_ext_forces_indep_v() {

    StackEntry stack_loc{};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    modules::SinkParticlesUpdate<Tvec, SPHKernel> sink_update(context, solver_config, storage);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(iaxyz_ext);
        field.field_raz();
    });

    sink_update.compute_sph_forces();

    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{buf_axyz_ext},
                    pdat.get_obj_cnt(),
                    [mGM = -cmass * G](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                        Tvec r_a       = xyz[gid];
                        Tscal abs_ra   = sycl::length(r_a);
                        Tscal abs_ra_3 = abs_ra * abs_ra * abs_ra;
                        axyz_ext[gid] += mGM * r_a / abs_ra_3;
                    });
            });

        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{buf_axyz_ext},
                    pdat.get_obj_cnt(),
                    [mGM = -cmass * G](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                        Tvec r_a       = xyz[gid];
                        Tscal abs_ra   = sycl::length(r_a);
                        Tscal abs_ra_3 = abs_ra * abs_ra * abs_ra;
                        axyz_ext[gid] += mGM * r_a / abs_ra_3;
                    });
            });
        } else if (
            EF_ShearingBoxForce *ext_force = std::get_if<EF_ShearingBoxForce>(&var_force.val)) {

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{buf_axyz_ext},
                    pdat.get_obj_cnt(),
                    [two_eta = 2 * ext_force->eta](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                        Tvec r_a = xyz[gid];
                        axyz_ext[gid] += Tvec{r_a.x() * two_eta, 0, 0};
                    });
            });

        } else {
            shambase::throw_unimplemented("this force is not handled, yet ...");
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::add_ext_forces() {

    StackEntry stack_loc{};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;

    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceBuffer<Tvec> &buf_axyz     = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

        sham::EventList depends_list;
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto axyz_ext = buf_axyz_ext.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(
                cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                    axyz[gid] += axyz_ext[gid];
                });
        });

        buf_axyz.complete_event_state(e);
        buf_axyz_ext.complete_event_state(e);
    });

    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_LenseThirring     = typename SolverConfigExtForce::LenseThirring;

    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {

        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {

            Tscal cmass = ext_force->central_mass;
            Tscal G     = solver_config.get_constant_G();
            Tscal c     = solver_config.get_constant_c();
            Tscal GM    = cmass * G;

            logger::raw_ln("S", ext_force->a_spin * GM * GM * ext_force->dir_spin / (c * c * c));

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
                sham::DeviceBuffer<Tvec> &buf_axyz = pdat.get_field_buf_ref<Tvec>(iaxyz);

                Tvec S = ext_force->a_spin * GM * GM * ext_force->dir_spin / (c * c * c);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz, buf_vxyz},
                    sham::MultiRef{buf_axyz},
                    pdat.get_obj_cnt(),
                    [S](u32 gid, const Tvec *xyz, const Tvec *vxyz, Tvec *axyz) {
                        Tvec r_a       = xyz[gid];
                        Tvec v_a       = vxyz[gid];
                        Tscal abs_ra   = sycl::length(r_a);
                        Tscal abs_ra_2 = abs_ra * abs_ra;
                        Tscal abs_ra_3 = abs_ra_2 * abs_ra;
                        Tscal abs_ra_5 = abs_ra_2 * abs_ra_2 * abs_ra;

                        Tvec omega_a
                            = (S * (2 / abs_ra_3)) - (6 * sham::dot(S, r_a) * r_a) / abs_ra_5;
                        Tvec acc_lt = sycl::cross(v_a, omega_a);
                        axyz[gid] += acc_lt;
                    });
            });
        } else if (
            EF_ShearingBoxForce *ext_force = std::get_if<EF_ShearingBoxForce>(&var_force.val)) {

            shamrock::patch::SimulationBoxInfo &sim_box = scheduler().get_sim_box();
            Tvec bsize                                  = sim_box.get_bounding_box_size<Tvec>();
            Tscal bsize_dir                             = bsize.x() * ext_force->shear_base.x()
                              + bsize.y() * ext_force->shear_base.y()
                              + bsize.z() * ext_force->shear_base.z();

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(0);
                sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
                sham::DeviceBuffer<Tvec> &buf_axyz = pdat.get_field_buf_ref<Tvec>(iaxyz);

                Tscal Omega_0    = ext_force->Omega_0;
                Tscal Omega_0_sq = Omega_0 * Omega_0;
                Tscal q_         = ext_force->q;

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz, buf_vxyz},
                    sham::MultiRef{buf_axyz},
                    pdat.get_obj_cnt(),
                    [Omega_0, Omega_0_sq, q = q_](
                        u32 gid, const Tvec *xyz, const Tvec *vxyz, Tvec *axyz) {
                        Tvec r_a = xyz[gid];
                        Tvec v_a = vxyz[gid];
                        axyz[gid] += Tvec{
                            2 * Omega_0 * (q * Omega_0 * r_a.x() + v_a.y()),
                            -2 * Omega_0 * v_a.x(),
                            -Omega_0_sq * r_a.z()};
                    });
            });

        } else {
            shambase::throw_unimplemented("this force is not handled, yet ...");
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::point_mass_accrete_particles() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_LenseThirring     = typename SolverConfigExtForce::LenseThirring;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    for (auto var_force : solver_config.ext_force_config.ext_forces) {

        Tvec pos_accretion;
        Tscal Racc;

        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {
            pos_accretion = {0, 0, 0};
            Racc          = ext_force->Racc;
        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {
            pos_accretion = {0, 0, 0};
            Racc          = ext_force->Racc;
        } else {
            continue;
        }

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

            sycl::buffer<u32> not_accreted(Nobj);
            sycl::buffer<u32> accreted(Nobj);

            sham::EventList depends_list;
            auto xyz = buf_xyz.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor not_acc{not_accreted, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor acc{accreted, cgh, sycl::write_only, sycl::no_init};

                Tvec r_sink    = pos_accretion;
                Tscal acc_rad2 = Racc * Racc;

                shambase::parallel_for(cgh, Nobj, "check accretion", [=](i32 id_a) {
                    Tvec r            = xyz[id_a] - r_sink;
                    bool not_accreted = sycl::dot(r, r) > acc_rad2;
                    not_acc[id_a]     = (not_accreted) ? 1 : 0;
                    acc[id_a]         = (!not_accreted) ? 1 : 0;
                });
            });

            buf_xyz.complete_event_state(e);

            std::tuple<std::optional<sycl::buffer<u32>>, u32> id_list_keep
                = shamalgs::numeric::stream_compact(q.q, not_accreted, Nobj);

            std::tuple<std::optional<sycl::buffer<u32>>, u32> id_list_accrete
                = shamalgs::numeric::stream_compact(q.q, accreted, Nobj);

            // sum accreted values onto sink

            if (std::get<1>(id_list_accrete) > 0) {

                u32 Naccrete = std::get<1>(id_list_accrete);

                Tscal acc_mass = gpart_mass * Naccrete;

                sycl::buffer<Tvec> pxyz_acc(Naccrete);

                sham::EventList depends_list;

                auto vxyz = buf_vxyz.get_read_access(depends_list);

                auto e = q.submit(depends_list, [&, gpart_mass](sycl::handler &cgh) {
                    sycl::accessor id_acc{*std::get<0>(id_list_accrete), cgh, sycl::read_only};

                    sycl::accessor accretion_p{pxyz_acc, cgh, sycl::write_only};

                    shambase::parallel_for(
                        cgh, Naccrete, "compute sum momentum accretion", [=](i32 id_a) {
                            accretion_p[id_a] = gpart_mass * vxyz[id_acc[id_a]];
                        });
                });

                buf_vxyz.complete_event_state(e);

                Tvec acc_pxyz = shamalgs::reduction::sum(q.q, pxyz_acc, 0, Naccrete);

                logger::raw_ln("central potential accretion : += ", acc_mass);

                pdat.keep_ids(*std::get<0>(id_list_keep), std::get<1>(id_list_keep));
            }
        });
    }
}

using namespace shammath;
template class shammodels::sph::modules::ExternalForces<f64_3, M4>;
template class shammodels::sph::modules::ExternalForces<f64_3, M6>;
template class shammodels::sph::modules::ExternalForces<f64_3, M8>;

template class shammodels::sph::modules::ExternalForces<f64_3, C2>;
template class shammodels::sph::modules::ExternalForces<f64_3, C4>;
template class shammodels::sph::modules::ExternalForces<f64_3, C6>;
