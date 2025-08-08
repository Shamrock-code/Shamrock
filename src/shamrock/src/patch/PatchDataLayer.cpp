// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchDataLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtree/kernels/geometry_utils.hpp"

namespace shamrock::patch {

    PatchDataLayer PatchDataLayer::mock_patchdata(
        u64 seed, u32 obj_cnt, const std::shared_ptr<PatchDataLayerLayout>&  pdl) {
        PatchDataLayer pdat{pdl};

        pdat.fields.clear();

        pdat.pdl().for_each_field_any([&](auto &field) {
            using f_t    = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            pdat.fields.push_back(
                var_t{PatchDataField<base_t>::mock_field(seed, obj_cnt, field.name, field.nvar)});
        });

        return pdat;
    }

    void PatchDataLayer::init_fields() {

        pdl().for_each_field_any([&](auto &field) {
            using f_t    = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            fields.push_back(var_t{PatchDataField<base_t>(field.name, field.nvar)});
        });
    }

    void PatchDataLayer::extract_element(u32 pidx, PatchDataLayer &out_pdat) {
        StackEntry stack_loc{};

        for (u32 idx = 0; idx < fields.size(); idx++) {

            std::visit(
                [&](auto &field, auto &out_field) {
                    using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                    using t2 =
                        typename std::remove_reference<decltype(out_field)>::type::Field_type;

                    if constexpr (std::is_same<t1, t2>::value) {
                        field.extract_element(pidx, out_field);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>("missmatch");
                    }
                },
                fields[idx].value,
                out_pdat.fields[idx].value);
        }
    }

    void PatchDataLayer::insert_elements(PatchDataLayer &pdat) {

        StackEntry stack_loc{};

        for (u32 idx = 0; idx < fields.size(); idx++) {

            std::visit(
                [&](auto &field, auto &out_field) {
                    using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                    using t2 =
                        typename std::remove_reference<decltype(out_field)>::type::Field_type;

                    if constexpr (std::is_same<t1, t2>::value) {
                        field.insert(out_field);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>("missmatch");
                    }
                },
                fields[idx].value,
                pdat.fields[idx].value);
        }
    }

    void PatchDataLayer::overwrite(PatchDataLayer &pdat, u32 obj_cnt) {
        StackEntry stack_loc{};

        for (u32 idx = 0; idx < fields.size(); idx++) {

            std::visit(
                [&](auto &field, auto &out_field) {
                    using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                    using t2 =
                        typename std::remove_reference<decltype(out_field)>::type::Field_type;

                    if constexpr (std::is_same<t1, t2>::value) {
                        field.overwrite(out_field, obj_cnt);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>("missmatch");
                    }
                },
                fields[idx].value,
                pdat.fields[idx].value);
        }
    }

    void PatchDataLayer::resize(u32 new_obj_cnt) {

        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.resize(new_obj_cnt);
            });
        }
    }

    void PatchDataLayer::reserve(u32 new_obj_cnt) {

        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.reserve(new_obj_cnt);
            });
        }
    }

    void PatchDataLayer::expand(u32 new_obj_cnt) {

        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.expand(new_obj_cnt);
            });
        }
    }

    void PatchDataLayer::index_remap(sycl::buffer<u32> &index_map, u32 len) {

        sham::DeviceBuffer<u32> dev_index_map(
            index_map, len, shamsys::instance::get_compute_scheduler_ptr());

        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.index_remap(dev_index_map, len);
            });
        }
    }

    void PatchDataLayer::index_remap_resize(sycl::buffer<u32> &index_map, u32 len) {
        sham::DeviceBuffer<u32> dev_index_map(
            index_map, len, shamsys::instance::get_compute_scheduler_ptr());

        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.index_remap_resize(dev_index_map, len);
            });
        }
    }

    void PatchDataLayer::index_remap_resize(sham::DeviceBuffer<u32> &index_map, u32 len) {
        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.index_remap_resize(index_map, len);
            });
        }
    }

    void PatchDataLayer::keep_ids(sycl::buffer<u32> &index_map, u32 len) {
        index_remap_resize(index_map, len);
    }

    void PatchDataLayer::keep_ids(sham::DeviceBuffer<u32> &index_map, u32 len) {
        index_remap_resize(index_map, len);
    }

    void PatchDataLayer::remove_ids(const sham::DeviceBuffer<u32> &indexes, u32 len) {
        for (auto &field_var : fields) {
            field_var.visit([&](auto &field) {
                field.remove_ids(indexes, len);
            });
        }
    }

    void PatchDataLayer::append_subset_to(sycl::buffer<u32> &idxs, u32 sz, PatchDataLayer &pdat) {
        StackEntry stack_loc{};

        for (u32 idx = 0; idx < fields.size(); idx++) {

            std::visit(
                [&](auto &field, auto &out_field) {
                    using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                    using t2 =
                        typename std::remove_reference<decltype(out_field)>::type::Field_type;

                    if constexpr (std::is_same<t1, t2>::value) {
                        field.append_subset_to(idxs, sz, out_field);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>("missmatch");
                    }
                },
                fields[idx].value,
                pdat.fields[idx].value);
        }
    }

    void PatchDataLayer::append_subset_to(std::vector<u32> &idxs, PatchDataLayer &pdat) {
        StackEntry stack_loc{};

        for (u32 idx = 0; idx < fields.size(); idx++) {

            std::visit(
                [&](auto &field, auto &out_field) {
                    using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                    using t2 =
                        typename std::remove_reference<decltype(out_field)>::type::Field_type;

                    if constexpr (std::is_same<t1, t2>::value) {
                        field.append_subset_to(idxs, out_field);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>("missmatch");
                    }
                },
                fields[idx].value,
                pdat.fields[idx].value);
        }
    }

    void PatchDataLayer::serialize_buf(shamalgs::SerializeHelper &serializer) {
        StackEntry stack_loc{};
        for_each_field_any([&](auto &f) {
            f.serialize_buf(serializer);
        });
    }

    shamalgs::SerializeSize PatchDataLayer::serialize_buf_byte_size() {
        shamalgs::SerializeSize sum{};
        for_each_field_any([&](auto &f) {
            sum += f.serialize_buf_byte_size();
        });
        return sum;
    }

    PatchDataLayer PatchDataLayer::deserialize_buf(
        shamalgs::SerializeHelper &serializer, std::shared_ptr<PatchDataLayerLayout> pdl) {
        StackEntry stack_loc{};

        return PatchDataLayer{
            pdl, [&](auto &pdat_fields) {
                pdl->for_each_field_any([&](auto &field) {
                    using f_t    = typename std::remove_reference<decltype(field)>::type;
                    using base_t = typename f_t::field_T;

                    pdat_fields.push_back(var_t{PatchDataField<base_t>::deserialize_buf(
                        serializer, field.name, field.nvar)});
                });
            }};
    }

    void PatchDataLayer::fields_raz() {
        for_each_field_any([&](auto &f) {
            f.field_raz();
        });
    }

    template<class T>
    void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<T, 8> min_box,
        std::array<T, 8> max_box) {

        StackEntry stack_loc{};

        PatchDataField<T> &main_field = fields[0].get_if_ref_throw<T>();

        auto get_vec_idx = [&](T vmin, T vmax) -> std::vector<u32> {
            return main_field.get_elements_with_range(
                [&](T val, T vmin, T vmax) {
                    return Patch::is_in_patch_converted(val, vmin, vmax);
                },
                vmin,
                vmax);
        };

        std::vector<u32> idx_p0 = get_vec_idx(min_box[0], max_box[0]);
        std::vector<u32> idx_p1 = get_vec_idx(min_box[1], max_box[1]);
        std::vector<u32> idx_p2 = get_vec_idx(min_box[2], max_box[2]);
        std::vector<u32> idx_p3 = get_vec_idx(min_box[3], max_box[3]);
        std::vector<u32> idx_p4 = get_vec_idx(min_box[4], max_box[4]);
        std::vector<u32> idx_p5 = get_vec_idx(min_box[5], max_box[5]);
        std::vector<u32> idx_p6 = get_vec_idx(min_box[6], max_box[6]);
        std::vector<u32> idx_p7 = get_vec_idx(min_box[7], max_box[7]);

        u32 el_cnt_new = idx_p0.size() + idx_p1.size() + idx_p2.size() + idx_p3.size()
                         + idx_p4.size() + idx_p5.size() + idx_p6.size() + idx_p7.size();

        if (get_obj_cnt() != el_cnt_new) {

            logger::err_ln(
                "PatchData",
                "error in patchdata split, the new element count doesn't match the old one");

            logger::err_ln("PatchData", min_box[0], max_box[0]);
            logger::err_ln("PatchData", min_box[1], max_box[1]);
            logger::err_ln("PatchData", min_box[2], max_box[2]);
            logger::err_ln("PatchData", min_box[3], max_box[3]);
            logger::err_ln("PatchData", min_box[4], max_box[4]);
            logger::err_ln("PatchData", min_box[5], max_box[5]);
            logger::err_ln("PatchData", min_box[6], max_box[6]);
            logger::err_ln("PatchData", min_box[7], max_box[7]);

            T vmin = sham::min(min_box[0], min_box[1]);
            vmin   = sham::min(vmin, min_box[2]);
            vmin   = sham::min(vmin, min_box[3]);
            vmin   = sham::min(vmin, min_box[4]);
            vmin   = sham::min(vmin, min_box[5]);
            vmin   = sham::min(vmin, min_box[6]);
            vmin   = sham::min(vmin, min_box[7]);

            T vmax = sham::max(max_box[0], max_box[1]);
            vmax   = sham::max(vmax, max_box[2]);
            vmax   = sham::max(vmax, max_box[3]);
            vmax   = sham::max(vmax, max_box[4]);
            vmax   = sham::max(vmax, max_box[5]);
            vmax   = sham::max(vmax, max_box[6]);
            vmax   = sham::max(vmax, max_box[7]);

            main_field.check_err_range(
                [&](T val, T vmin, T vmax) {
                    return Patch::is_in_patch_converted(val, vmin, vmax);
                },
                vmin,
                vmax);
        }

        // TODO create a extract subpatch function

        append_subset_to(idx_p0, pdats[0].get());
        append_subset_to(idx_p1, pdats[1].get());
        append_subset_to(idx_p2, pdats[2].get());
        append_subset_to(idx_p3, pdats[3].get());
        append_subset_to(idx_p4, pdats[4].get());
        append_subset_to(idx_p5, pdats[5].get());
        append_subset_to(idx_p6, pdats[6].get());
        append_subset_to(idx_p7, pdats[7].get());
    }

#ifndef DOXYGEN
    template void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<f32_3, 8> min_box,
        std::array<f32_3, 8> max_box);
    template void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<f64_3, 8> min_box,
        std::array<f64_3, 8> max_box);
    template void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<u32_3, 8> min_box,
        std::array<u32_3, 8> max_box);
    template void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<u64_3, 8> min_box,
        std::array<u64_3, 8> max_box);
    template void PatchDataLayer::split_patchdata(
        std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
        std::array<i64_3, 8> min_box,
        std::array<i64_3, 8> max_box);
#endif

} // namespace shamrock::patch
