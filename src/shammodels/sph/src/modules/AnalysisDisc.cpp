// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisDisc.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"
#include <shambackends/sycl.hpp>
#include <vector>

// typename shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::analysis_basis
template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_basis(
    Tscal Rmin, Tscal Rmax, u32 Nbin, const ShamrockCtx &context) -> analysis_basis {

    sham::DeviceBuffer<Tscal> bin_edges(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    bin_edges
        = shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::linspace(Rmin, Rmax, Nbin + 1);

    // get radius from xyz
    using namespace shamrock;
    using namespace shamrock::patch;

    // sham::EventList depends_list;
    // sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    // auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
    // });
    PatchScheduler &sched = shambase::get_check_ref(context.sched);
    PatchDataLayout &pdl  = scheduler().pdl;
    const u32 ixyz        = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz       = pdl.get_field_idx<Tvec>("vxyz");
    auto &merged_xyzh     = storage.merged_xyzh.get();

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();
    u64 Npart                                         = 0;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        Npart += pdat.get_obj_cnt();
    });
    sham::DeviceBuffer<Tscal> buf_radius(Npart, shamsys::instance::get_compute_scheduler_ptr());
    ;
    sham::DeviceBuffer<Tvec> buf_J(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jx(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jy(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jz(Npart, shamsys::instance::get_compute_scheduler_ptr());

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        u32 len                            = pdat.get_obj_cnt();
        MergedPatchData &merged_patch      = mpdat.get(cur_p.id_patch);
        PatchData &mpdat                   = merged_patch.pdat;
        sham::DeviceBuffer<Tvec> &buf_xyz  = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz);
        sham::DeviceQueue &q               = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_vxyz},
            sham::MultiRef{buf_radius, buf_J, buf_Jx, buf_Jy, buf_Jz},
            len,
            [](u32 i,
               const Tvec *__restrict xyz,
               const Tvec *__restrict vxyz,
               Tscal *__restrict buf_radius,
               Tvec *__restrict buf_J,
               Tscal *__restrict buf_Jx,
               Tscal *__restrict buf_Jy,
               Tscal *__restrict buf_Jz) {
                using namespace shamrock::sph;
                Tvec pos = xyz[i];
                Tvec vel = vxyz[i];

                Tscal radius = sycl::sqrt(sycl::dot(pos, pos));
                Tvec J       = sycl::cross(pos, vel); // @@@ pmass *
                Tscal Jx     = sycl::cross(pos, vel)[0];
                Tscal Jy     = sycl::cross(pos, vel)[1];
                Tscal Jz     = sycl::cross(pos, vel)[2];

                buf_radius[i] = radius;
                buf_J[i]      = J;
                buf_Jx[i]     = Jx;
                buf_Jy[i]     = Jy;
                buf_Jz[i]     = Jz;
            });
    });
    // end get radius from xyz
    // sham::DeviceBuffer<Tscal> buf_Jx = buf_J.template get_field_ref<Tscal>(0);
    u32 len                                          = buf_radius.get_size();
    shamalgs::numeric::histogram_result<Tscal> histo = shamalgs::numeric::device_histogram_full(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_radius, len);

    auto binned_Jx = shamalgs::numeric::binned_sum(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jx, buf_radius, Npart);

    auto binned_Jy = shamalgs::numeric::binned_sum(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jy, buf_radius, Npart);

    auto binned_Jz = shamalgs::numeric::binned_sum(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jz, buf_radius, Npart);

    sham::DeviceBuffer<Tscal> zmean(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> Sigma(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    return analysis_basis{
        std::move(histo.bins_center),
        std::move(histo.counts),
        std::move(binned_Jx),
        std::move(binned_Jy),
        std::move(binned_Jz),
        std::move(zmean),
        std::move(Sigma)};
}

// only 300 to 500 bins, we do that on the host !

//    std::vector<Tvec> buff_l_host(Nbin);
//    auto acc_J = basis.binned_J.copy_to_stdvec();
//
//    for (u32 i = 0; i < Nbin; i++) {
//            Tvec &l = buff_l_host[i];
//            Tvec &J = acc_J[i];
//            Tscal J_norm
//                = sycl::sqrt(sycl::dot(J, J));
//
//            l = l / J_norm; // @@@ add a check if J zero
//    }
//
//
//    return analysis_stage0{buff_l_host};
template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage0(
    analysis_basis &basis, u32 Nbin) -> analysis_stage0 {
    // still do it on device because data is there still

    sham::DeviceBuffer<Tscal> &buf_binned_Jx = basis.binned_Jx;
    sham::DeviceBuffer<Tscal> &buf_binned_Jy = basis.binned_Jy;
    sham::DeviceBuffer<Tscal> &buf_binned_Jz = basis.binned_Jz;
    sham::DeviceBuffer<Tvec> buf_binned_unit_J(
        Nbin, shamsys::instance::get_compute_scheduler_ptr());

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::kernel_call(
        q,
        sham::MultiRef{buf_binned_Jx, buf_binned_Jy, buf_binned_Jz},
        sham::MultiRef{buf_binned_unit_J},
        Nbin,
        [](u32 i,
           const Tscal *__restrict Jx,
           const Tscal *__restrict Jy,
           const Tscal *__restrict Jz,
           Tvec *__restrict unit_J) {
            Tscal J_norm = sycl::sqrt(Jx[i] * Jx[i] + Jy[i] * Jy[i] + Jz[i] * Jz[i]);
            unit_J[i]    = Tvec((Jx[i] / J_norm, Jy[i] / J_norm, Jz[i] / J_norm));
        });

    return analysis_stage0{std::move(buf_binned_unit_J)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage1(
    analysis_basis &basis, analysis_stage0 &stage0) -> analysis_stage1 {

    sham::DeviceBuffer<Tvec> &buf_binned_J_unit = stage0.unit_J;
    sham::DeviceBuffer<Tscal> &buf_radius       = basis.radius;

    sham::DeviceBuffer<Tscal> buf_tilt(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_twist(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_psi(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::kernel_call(
        q,
        sham::MultiRef{buf_radius, buf_binned_J_unit},
        sham::MultiRef{buf_tilt, buf_twist, buf_psi},
        Nbin,
        [&](u32 i,
            const Tscal *__restrict radius,
            const Tvec *__restrict J_unit,
            Tscal *__restrict tilt,
            Tscal *__restrict twist,
            Tscal *__restrict psi) {
            Tvec l   = J_unit[i];
            Tscal lx = l[0];
            Tscal ly = l[1];
            Tscal lz = l[2];
            Tscal r  = radius[i];

            *tilt  = 0.;
            *twist = 0.;
            *psi   = 0.;
            if (sycl::fabs(lz) > 0. && i < Nbin - 1) {
                *tilt  = r / lz; // @@@ missing a term
                *twist = shambase::constants::pi<Tscal> * 0.5 * sycl::atan(lx / lz);
            }

            if (i > 0 && i < Nbin - 1) {
                Tscal radius_diff = radius[i + 1] - radius[i - 1];
                Tscal psi_x       = (J_unit[i + 1][0] - J_unit[i - 1][0]) / radius_diff;
                Tscal psi_y       = (J_unit[i + 1][1] - J_unit[i - 1][1]) / radius_diff;
                Tscal psi_z       = (J_unit[i + 1][2] - J_unit[i - 1][2]) / radius_diff;
                *psi              = sycl::sqrt(psi_x * psi_x + psi_y * psi_y + psi_z * psi_z);
            }
        });

    return analysis_stage1{std::move(buf_tilt), std::move(buf_twist), std::move(buf_psi)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage2(
    analysis_stage1 &stage1) -> analysis_stage2 {
    sham::DeviceBuffer<Tscal> buff_H(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_H_on_R(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    return analysis_stage2{std::move(buff_H), std::move(buff_H_on_R)};
}

using namespace shammath;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M4>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M6>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M8>;
