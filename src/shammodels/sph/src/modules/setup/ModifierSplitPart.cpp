// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierSplitPart.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/setup/ModifierSplitPart.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec>
shamrock::patch::PatchDataLayer shammodels::sph::modules::ModifierSplitPart<Tvec>::next_n(
    u32 nmax) {

    if (n_split == 1) {
        return parent->next_n(nmax);
    }

    if (nmax >= n_split && nmax != 0) {
        nmax = nmax / n_split;
    }

    using namespace shamrock::patch;
    PatchScheduler &sched = shambase::get_check_ref(context.sched);
    auto dev_sched        = shamsys::instance::get_compute_scheduler_ptr();

    shamrock::patch::PatchDataLayer original_pdat = parent->next_n(nmax);
    u32 npart                                     = original_pdat.get_obj_cnt();

    u32 ixyz   = sched.pdl_old().get_field_idx<Tvec>("xyz");
    u32 ihpart = sched.pdl_old().get_field_idx<Tscal>("hpart");

    std::vector<Tvec> xyz    = original_pdat.get_field_buf_ref<Tvec>(ixyz).copy_to_stdvec();
    std::vector<Tscal> hpart = original_pdat.get_field_buf_ref<Tscal>(ihpart).copy_to_stdvec();

    PatchDataLayer tmp(sched.get_layout_ptr_old());

    // perform the split and insert
    for (u64 i = 0; i < n_split; i++) {
        shamrock::patch::PatchDataLayer tmp_pdat = original_pdat.duplicate();

        std::vector<u64> seeds = generator.next_n(npart);

        std::vector<Tvec> new_xyz(npart);

        for (u64 j = 0; j < npart; j++) {
            std::mt19937_64 eng(seeds[j]);
            new_xyz[j] = xyz[j] + hpart[j] * shamalgs::random::mock_gaussian_multidim<Tvec>(eng);
        }

        tmp_pdat.get_field<Tvec>(ixyz).override(new_xyz, npart);
        tmp.insert_elements(tmp_pdat);
    }

    // filter particles outside the box
    std::tuple<Tvec, Tvec> box = sched.get_box_volume<Tvec>();

    PatchDataField<Tvec> &xyz_field = tmp.get_field<Tvec>(ixyz);
    auto idx_to_remove              = xyz_field.get_ids_where(
        [bmin = std::get<0>(box), bmax = std::get<1>(box)](const Tvec *__restrict pos, u32 i) {
            return !Patch::is_in_patch_converted(pos[i], bmin, bmax);
        });
    tmp.remove_ids(idx_to_remove, idx_to_remove.get_size());

    // ammend smoothing length
    // See
    // https://github.com/danieljprice/phantom/blob/f6d5beea4db73f432bc2ca3eaa450320f2abee7a/src/utils/utils_splitmerge.f90#L64

    Tscal h_scaling_fact = sycl::pow(Tscal(n_split), -1. / 3.) * h_scaling;

    sham::DeviceBuffer<Tscal> &hpart_final = tmp.get_field_buf_ref<Tscal>(ihpart);
    sham::kernel_call(
        dev_sched->get_queue(),
        sham::MultiRef{},
        sham::MultiRef{hpart_final},
        hpart_final.get_size(),
        [h_scaling_fact](u32 i, Tscal *__restrict hpart) {
            hpart[i] *= h_scaling_fact;
        });

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierSplitPart<f64_3>;
