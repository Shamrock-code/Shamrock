// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AnalysisDisc.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/linspace.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/wrapper.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include <shambackends/sycl.hpp>
#include <numeric>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_basis(
    Tscal pmass, Tscal Rmin, Tscal Rmax, u32 Nbin, const ShamrockCtx &context) -> analysis_basis {

    sham::DeviceBuffer<Tscal> bin_edges(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    bin_edges = shamalgs::primitives::linspace(Rmin, Rmax, Nbin + 1);

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchScheduler &sched     = shambase::get_check_ref(context.sched);
    PatchDataLayerLayout &pdl = sched.pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

    // get Npart
    u64 Npart  = 0;
    u64 Npatch = 0;
    std::vector<u64> parts_per_patch{};
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, shamrock::scheduler::PatchData &pdat) {
        parts_per_patch.push_back(pdat.get_obj_cnt());
        Npart += pdat.get_obj_cnt();
        Npatch++;
    });

    std::vector<u32> patch_start_index(Npatch);
    std::exclusive_scan(
        parts_per_patch.begin(), parts_per_patch.end(), patch_start_index.begin(), 0);

    sham::DeviceBuffer<Tscal> buf_radius(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jx(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jy(Npart, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_Jz(Npart, shamsys::instance::get_compute_scheduler_ptr());

    u32 i = 0;
    scheduler().for_each_patchdata_nonempty(
        [&, this](Patch cur_p, shamrock::scheduler::PatchData &pdat) {
            u32 len                            = pdat.get_obj_cnt();
            u32 start_write                    = patch_start_index[i];
            sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::kernel_call(
                q,
                sham::MultiRef{buf_xyz, buf_vxyz},
                sham::MultiRef{buf_radius, buf_Jx, buf_Jy, buf_Jz},
                len,
                [pmass, start_write](
                    u32 i,
                    const Tvec *__restrict xyz,
                    const Tvec *__restrict vxyz,
                    Tscal *__restrict buf_radius,
                    Tscal *__restrict buf_Jx,
                    Tscal *__restrict buf_Jy,
                    Tscal *__restrict buf_Jz) {
                    using namespace shamrock::sph;
                    Tvec pos = xyz[i];
                    Tvec vel = vxyz[i];

                    Tscal radius = sycl::sqrt(sycl::dot(pos, pos));
                    Tscal Jx     = pmass * sycl::cross(pos, vel)[0];
                    Tscal Jy     = pmass * sycl::cross(pos, vel)[1];
                    Tscal Jz     = pmass * sycl::cross(pos, vel)[2];

                    buf_radius[i + start_write] = radius;
                    buf_Jx[i + start_write]     = Jx;
                    buf_Jy[i + start_write]     = Jy;
                    buf_Jz[i + start_write]     = Jz;
                });

            i++;
        });

    auto histo = shamalgs::numeric::device_histogram_full_mpi(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_radius, Npart);

    auto binned_Jx = shamalgs::numeric::binned_average_mpi(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jx, buf_radius, Npart);

    auto binned_Jy = shamalgs::numeric::binned_average_mpi(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jy, buf_radius, Npart);

    auto binned_Jz = shamalgs::numeric::binned_average_mpi(
        shamsys::instance::get_compute_scheduler_ptr(), bin_edges, Nbin, buf_Jz, buf_radius, Npart);

    auto buf_radius_key = buf_radius.copy();
    auto Sigma          = shamalgs::numeric::binned_compute<Tscal>(
        shamsys::instance::get_compute_scheduler_ptr(),
        bin_edges,
        Nbin,
        buf_radius,
        buf_radius_key,
        Npart,
        [pmass, Rmin, Rmax, Nbin](auto for_each_values, u32 bin_count) {
            Tscal sigma_bin    = 0;
            Tscal delta        = (Rmax - Rmin) / Nbin;
            Tscal delta_halfed = delta / 2;
            for_each_values([&](Tscal val) {
                Tscal area = shambase::constants::pi<Tscal>
                             * ((val + delta_halfed) * (val + delta_halfed)
                                - (val - delta_halfed) * (val - delta_halfed));
                sigma_bin += pmass / area;
            });
            return sigma_bin;
        });
    shamalgs::collective::reduce_buffer_in_place_sum(Sigma, MPI_COMM_WORLD);

    return analysis_basis{
        std::move(buf_radius),
        std::move(bin_edges),
        std::move(histo.bins_center),
        std::move(histo.counts),
        std::move(binned_Jx),
        std::move(binned_Jy),
        std::move(binned_Jz),
        std::move(Sigma)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage0(
    analysis_basis &basis, u32 Nbin) -> analysis_stage0 {
    // compute unit l
    // still do it on device because data is there still
    sham::DeviceBuffer<Tscal> &buf_binned_Jx = basis.binned_Jx;
    sham::DeviceBuffer<Tscal> &buf_binned_Jy = basis.binned_Jy;
    sham::DeviceBuffer<Tscal> &buf_binned_Jz = basis.binned_Jz;
    sham::DeviceBuffer<Tscal> buf_binned_lx(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_binned_ly(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_binned_lz(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::kernel_call(
        q,
        sham::MultiRef{buf_binned_Jx, buf_binned_Jy, buf_binned_Jz},
        sham::MultiRef{buf_binned_lx, buf_binned_ly, buf_binned_lz},
        Nbin,
        [](u32 i,
           const Tscal *__restrict Jx,
           const Tscal *__restrict Jy,
           const Tscal *__restrict Jz,
           Tscal *__restrict lx,
           Tscal *__restrict ly,
           Tscal *__restrict lz) {
            Tscal J_norm = sycl::sqrt(Jx[i] * Jx[i] + Jy[i] * Jy[i] + Jz[i] * Jz[i]);
            if (J_norm < shambase::get_epsilon<Tscal>()) {
                lx[i] = 0;
                ly[i] = 0;
                lz[i] = 0;
            } else {
                lx[i] = Jx[i] / J_norm;
                ly[i] = Jy[i] / J_norm;
                lz[i] = Jz[i] / J_norm;
            }
        });
    shamalgs::collective::reduce_buffer_in_place_sum(buf_binned_lx, MPI_COMM_WORLD);
    shamalgs::collective::reduce_buffer_in_place_sum(buf_binned_ly, MPI_COMM_WORLD);
    shamalgs::collective::reduce_buffer_in_place_sum(buf_binned_lz, MPI_COMM_WORLD);

    // compute zmean
    using namespace shamrock;
    using namespace shamrock::patch;
    PatchScheduler &sched     = shambase::get_check_ref(context.sched);
    PatchDataLayerLayout &pdl = sched.pdl();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

    // get Npart
    u64 Npart  = 0;
    u64 Npatch = 0;
    std::vector<u64> parts_per_patch{};
    // on all patches on all ranks, allreduce is included
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, shamrock::scheduler::PatchData &pdat) {
        parts_per_patch.push_back(pdat.get_obj_cnt());
        Npart += pdat.get_obj_cnt();
        Npatch++;
    });

    std::vector<u32> patch_start_index(Npatch);
    std::exclusive_scan(
        parts_per_patch.begin(), parts_per_patch.end(), patch_start_index.begin(), 0);

    u32 i = 0;
    sham::DeviceBuffer<Tscal> buf_zmean(Npart, shamsys::instance::get_compute_scheduler_ptr());
    // compute zmean: loop on all patches of all ranks
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, shamrock::scheduler::PatchData &pdat) {
        u32 len                            = pdat.get_obj_cnt();
        u32 start_write                    = patch_start_index[i];
        sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
        sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
        sham::DeviceQueue &q               = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{
                buf_xyz,
                basis.bin_edges,
                basis.buf_radius,
                buf_binned_lx,
                buf_binned_ly,
                buf_binned_lz},
            sham::MultiRef{buf_zmean},
            len,
            [this, Nbin, start_write](
                u32 i,
                const Tvec *__restrict xyz,
                const Tscal *__restrict bin_edges,
                const Tscal *__restrict radius,
                const Tscal *__restrict lx,
                const Tscal *__restrict ly,
                const Tscal *__restrict lz,
                Tscal *__restrict buf_zmean) {
                using namespace shamrock::sph;
                Tvec pos      = xyz[i];
                Tscal radiusi = radius[i];
                u32 bini      = mybin(radiusi, bin_edges, Nbin);

                // zdash
                buf_zmean[i + start_write]
                    = lx[bini] * pos[0] + ly[bini] * pos[1] + lz[bini] * pos[2];
            });

        i++;
    });

    // now that we have zmean, bin it
    auto binned_zmean = shamalgs::numeric::binned_average_mpi(
        shamsys::instance::get_compute_scheduler_ptr(),
        basis.bin_edges,
        Nbin,
        buf_zmean,
        basis.buf_radius,
        Npart);

    // now compute H
    u32 j = 0;
    sham::DeviceBuffer<Tscal> buf_H(Npart, shamsys::instance::get_compute_scheduler_ptr());
    // compute buf_H: loop on all patches of all ranks
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, shamrock::scheduler::PatchData &pdat) {
        u32 len                           = pdat.get_obj_cnt();
        u32 start_write                   = patch_start_index[j];
        sham::DeviceBuffer<Tvec> &buf_xyz = pdat.get_field_buf_ref<Tvec>(ixyz);
        sham::DeviceQueue &q              = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, basis.bin_edges, binned_zmean, buf_zmean},
            sham::MultiRef{buf_H},
            len,
            [this, Nbin, start_write](
                u32 i,
                const Tvec *__restrict xyz,
                const Tscal *__restrict bin_edges,
                const Tscal *__restrict binned_zmean,
                const Tscal *__restrict buf_zmean,
                Tscal *__restrict buf_H) {
                using namespace shamrock::sph;

                Tvec pos     = xyz[i];
                Tscal radius = sycl::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
                u32 bini     = mybin(radius, bin_edges, Nbin);

                buf_H[i + start_write]
                    = (buf_zmean[i] - binned_zmean[bini]) * (buf_zmean[i] - binned_zmean[bini]);
            });

        j++;
    });

    auto binned_Hsq = shamalgs::numeric::binned_average_mpi(
        shamsys::instance::get_compute_scheduler_ptr(),
        basis.bin_edges,
        Nbin,
        buf_H,
        basis.buf_radius,
        Npart);

    return analysis_stage0{
        std::move(buf_binned_lx),
        std::move(buf_binned_ly),
        std::move(buf_binned_lz),
        std::move(buf_zmean),
        std::move(binned_Hsq)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis_stage1(
    analysis_basis &basis, analysis_stage0 &stage0, u32 Nbin) -> analysis_stage1 {

    sham::DeviceBuffer<Tscal> &buf_lx     = stage0.lx;
    sham::DeviceBuffer<Tscal> &buf_ly     = stage0.ly;
    sham::DeviceBuffer<Tscal> &buf_lz     = stage0.lz;
    sham::DeviceBuffer<Tscal> &buf_radius = basis.radius;

    sham::DeviceBuffer<Tscal> buf_tilt(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_twist(Nbin, shamsys::instance::get_compute_scheduler_ptr());
    sham::DeviceBuffer<Tscal> buf_psi(Nbin, shamsys::instance::get_compute_scheduler_ptr());

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::kernel_call(
        q,
        sham::MultiRef{buf_radius, buf_lx, buf_ly, buf_lz},
        sham::MultiRef{buf_tilt, buf_twist, buf_psi},
        Nbin,
        [=](u32 i,
            const Tscal *__restrict radius,
            const Tscal *__restrict lx,
            const Tscal *__restrict ly,
            const Tscal *__restrict lz,
            Tscal *__restrict tilt,
            Tscal *__restrict twist,
            Tscal *__restrict psi) {
            Tscal r = radius[i];

            tilt[i]        = 0.;
            twist[i]       = 0.;
            psi[i]         = 0.;
            const u32 Nmax = Nbin - 1;
            if (sycl::fabs(lz[i]) > 0. && i < Nmax) {
                tilt[i]  = sycl::acos(lz[i]);         // @@@ same as phantom
                twist[i] = sycl::atan(ly[i] / lz[i]); // shambase::constants::pi<Tscal> * 0.5 *
            }

            if (i > 0 && i < Nmax) {
                Tscal radius_diff = radius[i + 1] - radius[i - 1];
                Tscal psi_x       = (lx[i + 1] - lx[i - 1]) / radius_diff;
                Tscal psi_y       = (ly[i + 1] - ly[i - 1]) / radius_diff;
                Tscal psi_z       = (lz[i + 1] - lz[i - 1]) / radius_diff;
                psi[i]            = sycl::sqrt(psi_x * psi_x + psi_y * psi_y + psi_z * psi_z);
            }
        });

    return analysis_stage1{std::move(buf_tilt), std::move(buf_twist), std::move(buf_psi)};
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>::compute_analysis(
    Tscal Rmin, Tscal Rmax, u32 Nbin, const ShamrockCtx &ctx) -> analysis {

    const Tscal pmass      = solver_config.gpart_mass;
    analysis_basis basis   = compute_analysis_basis(pmass, Rmin, Rmax, Nbin, ctx);
    analysis_stage0 stage0 = compute_analysis_stage0(basis, Nbin);
    analysis_stage1 stage1 = compute_analysis_stage1(basis, stage0, Nbin);

    return analysis{
        std::move(basis.radius),
        std::move(basis.counter),
        std::move(basis.Sigma),
        std::move(stage0.lx),
        std::move(stage0.ly),
        std::move(stage0.lz),
        std::move(stage1.tilt),
        std::move(stage1.twist),
        std::move(stage1.psi),
        std::move(stage0.Hsq)};
}

using namespace shammath;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M4>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M6>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, M8>;

template class shammodels::sph::modules::AnalysisDisc<f64_3, C2>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, C4>;
template class shammodels::sph::modules::AnalysisDisc<f64_3, C6>;
