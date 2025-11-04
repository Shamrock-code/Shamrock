// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sphpatch.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/legacy/patch/base/patchdata.hpp"
// #include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/legacy/utils/syclreduction.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include <type_traits>
#include <stdexcept>

namespace patchdata {
    namespace sph {

        template<class htype>
        inline htype get_h_max(
            shamrock::patch::PatchDataLayerLayout &pdl,
            sycl::queue &queue,
            shamrock::patch::PatchDataLayer &pdat) {

            if (pdat.get_obj_cnt() == 0)
                return 0;

            htype tmp;

            u32 nobj = pdat.get_obj_cnt();

            if constexpr (std::is_same<htype, f32>::value) {

                u32 ihpart = pdl.get_field_idx<f32>("hpart");
                tmp        = syclalg::get_max<f32>(pdat.get_field<f32>(ihpart).get_buf(), nobj);

            } else if constexpr (std::is_same<htype, f64>::value) {
                u32 ihpart = pdl.get_field_idx<f64>("hpart");
                tmp        = syclalg::get_max<f64>(pdat.get_field<f64>(ihpart).get_buf(), nobj);

            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "get_h_max -> current htype not handled");
            }

            return tmp;
        }

#if false

        template<class htype>
        [[deprecated]]
        inline htype get_h_max(PatchDataLayerLayout & pdl,sycl::queue & queue, PatchDataBuffer & pdatbuf){

            if(pdatbuf.element_count == 0) return 0;

            htype tmp;


            u32 & nobj = pdatbuf.element_count;

            if constexpr (std::is_same<htype, f32>::value){

                u32 ihpart = pdl.get_field_idx<f32>(::sph::field_names::field_hpart);
                tmp = syclalg::get_max<f32>(queue, pdatbuf.fields_f32[ihpart],nobj);

            } else if constexpr (std::is_same<htype, f64>::value){
                u32 ihpart = pdl.get_field_idx<f64>(::sph::field_names::field_hpart);
                tmp = syclalg::get_max<f64>(queue, pdatbuf.fields_f64[ihpart],nobj);

            }else{
                throw shamrock_exc("get_h_max -> current htype not handled");
            }

            return tmp;

        }



        template<class vec>
        [[deprecated]]
        inline std::tuple<vec,vec> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf);

        template<>
        inline std::tuple<f32_3,f32_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){

            u32 ihpart = pdatbuf.pdl.get_field_idx<f32_3>("xyz");
            return syclalg::get_min_max<f32_3>(queue, pdatbuf.fields_f32_3[ihpart],pdatbuf.element_count);

        }

        template<>
        inline std::tuple<f64_3,f64_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){
            u32 ihpart = pdatbuf.pdl.get_field_idx<f64_3>("xyz");
            return syclalg::get_min_max<f64_3>(queue, pdatbuf.fields_f64_3[ihpart],pdatbuf.element_count);

        }

#endif

    } // namespace sph
} // namespace patchdata
