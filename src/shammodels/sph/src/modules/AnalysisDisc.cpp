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
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec>
Tvec linspace(double Rmin, double Rmax, int N) {
    Tvec bins(N);
    double step = (Rmax - Rmin) / (N - 1);
    for (int i = 0; i < N; ++i) {
        bins[i] = Rmin + i * step;
    }
    return bins;
}
// typename shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::analysis_basis
template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_basis(
    Tscal Rmin, Tscal Rmax, u32 Nbin, const sham::DeviceScheduler_ptr &sched) {

    Tvec bin_edges = linspace(Rmin, Rmax, Nbin + 1);

    // get radius from xyz
    sham::DeviceBuffer<Tscal> buf_radius;
    sham::DeviceBuffer<Tvec> buf_J;
    using namespace shamrock;
    using namespace shamrock::patch;

    // sham::EventList depends_list;
    // sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    // auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
    // });

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");
    auto &merged_xyzh    = storage.merged_xyzh.get();

    shambase::DistributedData<MergedPatchData> &mpdat = storage.merged_patchdata_ghost.get();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        u32 len                            = pdat.get_obj_cnt();
        MergedPatchData &merged_patch      = mpdat.get(cur_p.id_patch);
        PatchData &mpdat                   = merged_patch.pdat;
        sham::DeviceBuffer<Tvec> &buf_xyz  = merged_xyzh.get(cur_p.id_patch).field_pos.get_buf();
        sham::DeviceBuffer<Tvec> &buf_vxyz = mpdat.get_field_buf_ref<Tvec>(ivxyz);
        tree::ObjectCache &pcache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(cur_p.id_patch);
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_vxyz},
            sham::MultiRef{buf_radius, buf_J},
            len,
            [](u32 i,
               const Tvec *__restrict xyz,
               const Tvec *__restrict vxyz,
               Tscal *__restrict buf_radius,
               Tvec *__restrict buf_J) {
                using namespace shamrock::sph;
                Tvec pos = xyz[i];
                Tvec vel = vxyz[i];

                Tscal radius = sycl::sqrt(sycl::dot(pos, pos));
                Tvec J       = sycl::sqrt(sycl::cross(pos, vel));

                buf_radius[i] = radius;
                buf_J[i]      = J;
            });
    });
    // end get radius from xyz

    u64 len = buf_radius.size();
    shamalgs::numeric::histogram_result<Tvec> histo
        = device_histogram_full(sched, bin_edges, Nbin, buf_radius, len);

    auto position = merged_xyzh.get().field_pos.get_buf();

    auto binned_J = shamalgs::numeric::binned_compute(
        sched,
        bin_edges,
        Nbin,
        buf_J,
        position,
        position.get_size(),
        [](auto for_each_values, u32 bin_count) {
            Tvec sum = Tvec::Zero();
            for_each_values([&](Tvec v) {
                sum += v;
            });
            return sum;
        });

    sham::DeviceBuffer<Tscal> zmean;
    sham::DeviceBuffer<Tscal> Sigma;

    return analysis_basis{histo.bins_center, histo.counts, binned_J, zmean, Sigma};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage0(
    analysis_basis &basis) -> analysis_stage0 {

    sham::DeviceBuffer<Tscal> buff_lx(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_ly(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_lz(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_lx = buff_lx.get_write_access(depends_list);
    auto acc_ly = buff_ly.get_write_access(depends_list);
    auto acc_lz = buff_lz.get_write_access(depends_list);

    auto acc_Jx = basis.Jx.copy_to_stdvec();
    auto acc_Jy = basis.Jy.copy_to_stdvec();
    auto acc_Jz = basis.Jz.copy_to_stdvec();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        for (u32 i = 0; i < Nbin; i++) {
            Tscal &lx = acc_lx[i];
            Tscal &ly = acc_ly[i];
            Tscal &lz = acc_lz[i];

            Tscal J_norm
                = sycl::sqrt(acc_Jx[i] * acc_Jx[i] + acc_Jy[i] * acc_Jy[i] + acc_Jz[i] * acc_Jz[i]);

            lx = acc_Jx[i] / J_norm;
            ly = acc_Jy[i] / J_norm;
            lz = acc_Jz[i] / J_norm;
        }
    });

    return analysis_stage0{std::move(buff_lx), std::move(buff_ly), std::move(buff_lz)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage1(
    analysis_basis &basis, analysis_stage0 &stage0) -> analysis_stage1 {

    sham::DeviceBuffer<Tscal> buff_tilt(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_twist(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buff_psi(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_tilt  = buff_tilt.get_write_access(depends_list); // @@@ necessary on device ?
    auto acc_twist = buff_twist.get_write_access(depends_list);
    auto acc_psi   = buff_psi.get_write_access(depends_list);

    auto acc_radius = basis.radius.copy_to_stdvec();
    auto acc_lx     = stage0.lx.copy_to_stdvec();
    auto acc_ly     = stage0.ly.copy_to_stdvec();
    auto acc_lz     = stage0.lz.copy_to_stdvec();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        for (u32 i = 0; i < Nbin; i++) {
            Tscal radius = acc_radius[i];
            Tscal lx     = acc_lx[i];
            Tscal ly     = acc_ly[i];
            Tscal lz     = acc_lz[i];

            Tscal &tilt  = acc_tilt[i];
            Tscal &twist = acc_twist[i];
            Tscal &psi   = acc_psi[i];

            if (sycl::fabs(lz) > 0. && i < Nbin - 1) {
                tilt  = radius / lz; // @@@ missing a term
                twist = shambase::constants::pi<Tscal> * 0.5 * sycl::atan(lx / lz);
            }

            if (i > 0 && i < Nbin - 1) {
                Tscal radius_diff = acc_radius[i + 1] - acc_radius[i - 1];
                Tscal psi_x       = (acc_lx[i + 1] - acc_lx[i - 1]) / radius_diff;
                Tscal psi_y       = (acc_ly[i + 1] - acc_ly[i - 1]) / radius_diff;
                Tscal psi_z       = (acc_lz[i + 1] - acc_lz[i - 1]) / radius_diff;
                psi               = sycl::sqrt(psi_x * psi_x + psi_y * psi_y + psi_z * psi_z);
            }
        }
    });

    buff_tilt.complete_event_state(e);
    buff_twist.complete_event_state(e);
    buff_psi.complete_event_state(e);

    return analysis_stage1{std::move(buff_tilt), std::move(buff_twist), std::move(buff_psi)};
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
