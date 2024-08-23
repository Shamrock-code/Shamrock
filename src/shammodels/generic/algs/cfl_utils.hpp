// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file cfl_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/legacy/utils/syclreduction.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"

//%Impl status : Good

template<class flt>
class CflUtility {
    public:
    /*
    template<class LambdaCFL>
    inline static flt basic_cfl(PatchDataBuffer &pdat_buf,LambdaCFL && lambda_internal){



        u32 npart_patch = pdat_buf.element_count;

        std::unique_ptr<sycl::buffer<flt>> buf_cfl =
    std::make_unique<sycl::buffer<flt>>(npart_patch);

        sycl::range<1> range_npart{npart_patch};

        auto ker_reduc_step_mincfl = [&](sycl::handler &cgh) {
            auto arr = buf_cfl->template get_access<sycl::access::mode::discard_write>(cgh);

            lambda_internal(cgh,*buf_cfl, range_npart);
        };

        shamsys::instance::get_compute_queue().submit(ker_reduc_step_mincfl);

        flt min_cfl = syclalg::get_min<flt>(shamsys::instance::get_compute_queue(), buf_cfl);

        return min_cfl;

    }
    */

    template<class LambdaCFL>
    inline static flt basic_cfl(shamrock::patch::PatchData &pdat, LambdaCFL &&lambda_internal) {

        u32 npart_patch = pdat.get_obj_cnt();

        std::unique_ptr<sycl::buffer<flt>> buf_cfl
            = std::make_unique<sycl::buffer<flt>>(npart_patch);

        sycl::range<1> range_npart{npart_patch};

        auto ker_reduc_step_mincfl = [&](sycl::handler &cgh) {
            auto arr = buf_cfl->template get_access<sycl::access::mode::discard_write>(cgh);

            lambda_internal(cgh, *buf_cfl, range_npart);
        };

        shamsys::instance::get_compute_queue().submit(ker_reduc_step_mincfl);

        flt min_cfl
            = syclalg::get_min<flt>(shamsys::instance::get_compute_queue(), buf_cfl, npart_patch);

        return min_cfl;
    }
};
