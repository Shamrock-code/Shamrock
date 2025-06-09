// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonCodeSet.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include <stdexcept>
#include <string>
#include <utility>

namespace shamtree {

    namespace details {
        template<class Tmorton, class Tvec, u32 dim>
        class MortonKernelsUtils {

            public:
            using Morton        = shamrock::sfc::MortonCodes<Tmorton, dim>;
            using MortonConvert = shamrock::sfc::MortonConverter<Tmorton, Tvec, dim>;

            using pos_t   = Tvec;
            using coord_t = typename shambase::VectorProperties<pos_t>::component_type;
            using ipos_t  = typename Morton::int_vec_repr;
            using int_t   = typename Morton::int_vec_repr_base;

            using CoordTransform
                = shammath::CoordRangeTransform<typename Morton::int_vec_repr, Tvec>;

            inline static CoordTransform
            get_transform(pos_t bounding_box_min, pos_t bounding_box_max) {
                return MortonConvert::get_transform(bounding_box_min, bounding_box_max);
            }
            inline static ipos_t to_morton_grid(pos_t pos, CoordTransform transform) {
                return MortonConvert::to_morton_grid(pos, transform);
            }

            inline static pos_t to_real_space(ipos_t pos, CoordTransform transform) {
                return MortonConvert::to_real_space(pos, transform);
            }
        };

    } // namespace details

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSet<Tmorton, Tvec, dim> morton_code_set_from_positions(
        const sham::DeviceScheduler_ptr &dev_sched,
        shammath::AABB<Tvec> bounding_box,
        sham::DeviceBuffer<Tvec> &pos_buf,
        u32 cnt_obj,
        u32 morton_count,
        sham::DeviceBuffer<Tmorton> &&cache_buf_morton_codes) {

        sham::DeviceBuffer<Tmorton> morton_codes
            = std::forward<sham::DeviceBuffer<Tmorton>>(cache_buf_morton_codes);

        morton_codes.resize(morton_count);

        if (morton_count < cnt_obj) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "MortonCodeSet: morton_count < cnt_obj\n morton_count: {}, cnt_obj: {}",
                morton_count,
                cnt_obj));
        }

        using Utils = details::MortonKernelsUtils<Tmorton, Tvec, dim>;

        auto transform = Utils::get_transform(bounding_box.lower, bounding_box.upper);

        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{pos_buf},
            sham::MultiRef{morton_codes},
            cnt_obj,
            [transf = transform,
             bb     = bounding_box](u32 i, const Tvec *__restrict pos, Tmorton *__restrict morton) {
                Tvec r     = pos[i];
                auto m     = Utils::to_morton_grid(bb.clamp_coord(r), transf);
                auto mcode = Utils::Morton::icoord_to_morton(m.x(), m.y(), m.z());

                morton[i] = mcode;
            });

        if (morton_count > cnt_obj) {
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{morton_codes},
                morton_count - cnt_obj,
                [err_code = shamrock::sfc::MortonInfo<Tmorton>::err_code,
                 cnt_obj  = cnt_obj](u32 i, Tmorton *__restrict morton) {
                    morton[cnt_obj + i] = err_code;
                });
        }

        return MortonCodeSet<Tmorton, Tvec, dim>(
            std::move(bounding_box),
            std::move(cnt_obj),
            std::move(morton_count),
            std::move(morton_codes));
    }

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSet<Tmorton, Tvec, dim> morton_code_set_from_positions(
        const sham::DeviceScheduler_ptr &dev_sched,
        shammath::AABB<Tvec> bounding_box,
        sham::DeviceBuffer<Tvec> &pos_buf,
        u32 cnt_obj,
        u32 morton_count) {

        sham::DeviceBuffer<Tmorton> morton_codes(morton_count, dev_sched);

        return morton_code_set_from_positions<Tmorton, Tvec, dim>(
            dev_sched, bounding_box, pos_buf, cnt_obj, morton_count, std::move(morton_codes));
    }

} // namespace shamtree

template class shamtree::MortonCodeSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSet<u64, f64_3, 3>;

template shamtree::MortonCodeSet<u32, f64_3, 3> shamtree::morton_code_set_from_positions(
    const sham::DeviceScheduler_ptr &dev_sched,
    shammath::AABB<f64_3> bounding_box,
    sham::DeviceBuffer<f64_3> &pos_buf,
    u32 cnt_obj,
    u32 morton_count);

template shamtree::MortonCodeSet<u64, f64_3, 3> shamtree::morton_code_set_from_positions(
    const sham::DeviceScheduler_ptr &dev_sched,
    shammath::AABB<f64_3> bounding_box,
    sham::DeviceBuffer<f64_3> &pos_buf,
    u32 cnt_obj,
    u32 morton_count);
