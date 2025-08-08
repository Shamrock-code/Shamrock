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
 * @file PatchDataLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "Patch.hpp"
#include "PatchDataField.hpp"
#include "PatchDataLayerLayout.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/intervals.hpp"
#include <variant>
#include <vector>

namespace shamrock::patch {
    /**
     * @brief PatchDataLayer container class, the layout is described in patchdata_layout
     */
    class PatchDataLayer {

        void init_fields();

        using var_t = FieldVariant<PatchDataField>;

        std::vector<var_t> fields;
        std::shared_ptr<PatchDataLayerLayout> pdl_ptr;

        public:
        using field_variant_t = var_t;

        inline PatchDataLayerLayout & pdl(){
            return shambase::get_check_ref(pdl_ptr);
        }

        inline std::shared_ptr<PatchDataLayerLayout> get_layout_ptr() const {
            return pdl_ptr;
        }

        inline PatchDataLayer(std::shared_ptr<PatchDataLayerLayout> pdl) : pdl_ptr(pdl) {
            init_fields();
        }

        inline PatchDataLayer(const PatchDataLayer &other) : pdl_ptr(other.get_layout_ptr()) {

            NamedStackEntry stack_loc{"PatchDataLayer::copy_constructor", true};

            for (auto &field_var : other.fields) {

                field_var.visit([&](auto &field) {
                    using base_t =
                        typename std::remove_reference<decltype(field)>::type::Field_type;
                    fields.emplace_back(PatchDataField<base_t>(field));
                });
            };
        }

        /**
         * @brief PatchDataLayer move constructor
         *
         * @param other
         */
        inline PatchDataLayer(PatchDataLayer &&other) noexcept
            : fields(std::move(other.fields)), pdl_ptr(std::move(other.pdl_ptr)) {
        }

        /**
         * @brief PatchDataLayer move assignment
         *
         * @param other
         */
        inline PatchDataLayer &operator=(PatchDataLayer &&other) noexcept {
            fields = std::move(other.fields);
            pdl_ptr    = std::move(other.pdl_ptr);
            return *this;
        }

        PatchDataLayer &operator=(const PatchDataLayer &other) = delete;

        static PatchDataLayer mock_patchdata(u64 seed, u32 obj_cnt, PatchDataLayerLayout &pdl);

        template<class Functor>
        inline void for_each_field_any(Functor &&func) {
            for (auto &f : fields) {
                f.visit([&](auto &arg) {
                    func(arg);
                });
            }
        }

        template<class Func>
        inline PatchDataLayer(std::shared_ptr<PatchDataLayerLayout> pdl, Func &&fct_init)
            : pdl_ptr(pdl) {

            
            u32 cnt = 0;

            fct_init(fields);
        }

        inline PatchDataLayer duplicate() {
            const PatchDataLayer &current = *this;
            return PatchDataLayer(current);
        }

        inline std::unique_ptr<PatchDataLayer> duplicate_to_ptr() {
            const PatchDataLayer &current = *this;
            return std::make_unique<PatchDataLayer>(current);
        }

        /**
         * @brief extract particle at index pidx and insert it in the provided vectors
         *
         * @param pidx
         * @param out_pdat
         */
        void extract_element(u32 pidx, PatchDataLayer &out_pdat);

        void keep_ids(sycl::buffer<u32> &index_map, u32 len);

        void insert_elements(PatchDataLayer &pdat);

        /**
         * @brief insert elements of pdat only if they are within the range
         *
         * @tparam T
         * @param pdat
         * @param bmin
         * @param bmax
         */
        template<class T>
        void insert_elements_in_range(PatchDataLayer &pdat, T bmin, T bmax);

        void resize(u32 new_obj_cnt);

        void reserve(u32 new_obj_cnt);

        void expand(u32 obj_cnt);

        /**
         * @brief this function remaps the patchdatafield like so
         *   val[id] = val[index_map[id]]
         * This function can be used to apply the result of a sort to the field
         *
         * @param index_map
         * @param len the length of the map (must match with the current count)
         */
        void index_remap(sycl::buffer<u32> &index_map, u32 len);

        /**
         * @brief this function remaps the patchdatafield like so
         *   val[id] = val[index_map[id]]
         * This function can be used to apply the result of a sort to the field
         *
         * @param index_map
         * @param len the length of the map
         */
        void index_remap_resize(sycl::buffer<u32> &index_map, u32 len);

        /// Same as index_remap_resize with a shamrock device buffer instead
        void index_remap_resize(sham::DeviceBuffer<u32> &index_map, u32 len);

        /// Same as keep_ids with a shamrock device buffer instead
        void keep_ids(sham::DeviceBuffer<u32> &index_map, u32 len);

        /// remove some particles ids
        void remove_ids(const sham::DeviceBuffer<u32> &indexes, u32 len);

        // template<class Tvecbox>
        // void split_patchdata(PatchDataLayer & pd0,PatchDataLayer & pd1,PatchDataLayer &
        // pd2,PatchDataLayer & pd3,PatchDataLayer & pd4,PatchDataLayer & pd5,PatchDataLayer &
        // pd6,PatchDataLayer & pd7,
        //     Tvecbox bmin_p0,Tvecbox bmin_p1,Tvecbox bmin_p2,Tvecbox bmin_p3,Tvecbox
        //     bmin_p4,Tvecbox bmin_p5,Tvecbox bmin_p6,Tvecbox bmin_p7, Tvecbox bmax_p0,Tvecbox
        //     bmax_p1,Tvecbox bmax_p2,Tvecbox bmax_p3,Tvecbox bmax_p4,Tvecbox bmax_p5,Tvecbox
        //     bmax_p6,Tvecbox bmax_p7);

        template<class Tvecbox>
        void split_patchdata(
            std::array<std::reference_wrapper<PatchDataLayer>, 8> pdats,
            std::array<Tvecbox, 8> min_box,
            std::array<Tvecbox, 8> max_box);

        void append_subset_to(std::vector<u32> &idxs, PatchDataLayer &pdat);
        void append_subset_to(sycl::buffer<u32> &idxs, u32 sz, PatchDataLayer &pdat);

        inline u32 get_obj_cnt() {

            bool is_empty = fields.empty();

            if (!is_empty) {
                return fields[0].visit_return([](auto &field) {
                    return field.get_obj_cnt();
                });
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "this PatchDataLayer does not contain any fields");
        }

        inline u64 memsize() {
            u64 sum = 0;

            for (auto &field_var : fields) {

                field_var.visit([&](auto &field) {
                    sum += field.memsize();
                });
            }

            return sum;
        }

        inline bool is_empty() { return get_obj_cnt() == 0; }

        void synchronize_buf() {
            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    field.synchronize_buf();
                });
            }
        }

        void overwrite(PatchDataLayer &pdat, u32 obj_cnt);

        template<class T>
        bool check_field_type(u32 idx) {
            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return true;
            } else {
                return false;
            }
        }

        template<class T>
        PatchDataField<T> &get_field(u32 idx) {

            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return *pval;
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "the request id is not of correct type\n"
                "   current map is : \n"
                + pdl().get_description_str()
                + "\n"
                  "    arg : idx = "
                + std::to_string(idx));
        }

        template<class T>
        sham::DeviceBuffer<T> &get_field_buf_ref(u32 idx) {

            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return pval->get_buf();
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "the request id is not of correct type\n"
                "   current map is : \n"
                + pdl().get_description_str()
                + "\n"
                  "    arg : idx = "
                + std::to_string(idx));
        }

        /**
         * @brief returns a PatchDataFieldSpan of the field at index idx, with the given nvar value
         *
         * @param idx the index of the field
         * @return a PatchDataFieldSpan
         */
        template<class T, u32 nvar>
        PatchDataFieldSpan<T, nvar> get_field_span(u32 idx) {
            return get_field<T>(idx).template get_span<nvar>();
        }

        /**
         * @brief returns a PatchDataFieldSpan of the field at index idx, with a dynamic number of
         * variables
         *
         * @param idx the index of the field
         * @return a PatchDataFieldSpan
         */
        template<class T>
        PatchDataFieldSpan<T, shamrock::dynamic_nvar> get_field_span_nvar_dynamic(u32 idx) {
            return get_field<T>(idx).get_span_nvar_dynamic();
        }

        template<class T>
        PatchDataFieldSpan<T, shamrock::dynamic_nvar, shamrock::access_t_pointer>
        get_field_pointer_span(u32 idx) {
            return get_field<T>(idx).get_pointer_span();
        }

        /**
         * @brief check that all contained field have the same obj cnt
         *
         */
        inline void check_field_obj_cnt_match() {
            u32 cnt = get_obj_cnt();
            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.get_obj_cnt() != cnt) {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "mismatch in obj cnt");
                    }
                });
            }
        }

        // template<class T> inline std::vector<PatchDataField<T> & > get_field_list(){
        //     std::vector<PatchDataField<T> & > ret;
        //
        //
        //}

        template<class T, class Functor>
        inline void for_each_field(Functor &&func) {
            for (auto &f : fields) {
                PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&f.value);

                if (pval) {
                    func(*pval);
                }
            }
        }

        inline friend bool operator==(PatchDataLayer &p1, PatchDataLayer &p2) {
            bool check = true;

            if (p1.fields.size() != p2.fields.size()) {
                return false;
            }

            for (u32 idx = 0; idx < p1.fields.size(); idx++) {

                bool ret = std::visit(
                    [&](auto &pf1, auto &pf2) -> bool {
                        using t1 = typename std::remove_reference<decltype(pf1)>::type::Field_type;
                        using t2 = typename std::remove_reference<decltype(pf2)>::type::Field_type;

                        if constexpr (std::is_same<t1, t2>::value) {
                            return pf1.check_field_match(pf2);
                        } else {
                            return false;
                        }
                    },
                    p1.fields[idx].value,
                    p2.fields[idx].value);

                check = check && ret;
            }

            return check;
        }

        void serialize_buf(shamalgs::SerializeHelper &serializer);

        shamalgs::SerializeSize serialize_buf_byte_size();

        static PatchDataLayer
        deserialize_buf(shamalgs::SerializeHelper &serializer, PatchDataLayerLayout &pdl);

        void fields_raz();

        bool has_nan() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_nan()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }
        bool has_inf() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_inf()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }
        bool has_nan_or_inf() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_nan_or_inf()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }

        /**
         * @brief
         * \todo should add a check in patch data to check that
         * size in ovveride match with the one in the input vec
         * @tparam T
         * @param field_name
         * @param vec
         */
        template<class T>
        void override_patch_field(std::string field_name, std::vector<T> &vec) {
            u32 len              = vec.size();
            PatchDataField<T> &f = get_field<T>(pdl().get_field_idx<T>(field_name));
            sycl::buffer<T> buf(vec.data(), len);
            f.override(buf, len);
        }

        /**
         * @brief Fetch data of a patchdata field into a std::vector
         *
         * @todo Improve for nvar != 1
         *
         * @tparam T
         * @param key
         * @param pdat
         * @return std::vector<T>
         */
        template<class T>
        inline std::vector<T> fetch_data(std::string key) {

            std::vector<T> vec;

            auto appender = [&](auto &field) {
                if (field.get_name() == key) {

                    shamlog_debug_ln("PyShamrockCTX", "appending field", key);

                    if (!field.is_empty()) {
                        auto acc = field.get_buf().copy_to_stdvec();
                        u32 len  = field.get_val_cnt();

                        for (u32 i = 0; i < len; i++) {
                            vec.push_back(acc[i]);
                        }
                    }
                }
            };

            for_each_field<T>([&](auto &field) {
                appender(field);
            });

            return vec;
        }
    };

    template<class T>
    inline void PatchDataLayer::insert_elements_in_range(PatchDataLayer &pdat, T bmin, T bmax) {

        StackEntry stack_loc{};

        if (!pdl().check_main_field_type<T>()) {

            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the chosen type for the main field does not match the required template type");
        }

        PatchDataField<T> &main_field = pdat.get_field<T>(0);

        auto get_vec_idx = [&](T vmin, T vmax) -> std::vector<u32> {
            return main_field.get_elements_with_range(
                [&](T val, T vmin, T vmax) {
                    if (shambase::VectorProperties<T>::dimension == 3) {
                        return shammath::is_in_half_open(val, vmin, vmax);
                    } else {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "dimension != 3 is not handled");
                    }
                },
                vmin,
                vmax);
        };

        std::vector<u32> idx_lst = get_vec_idx(bmin, bmax);

        shamlog_debug_sycl_ln("PatchDataLayer", "inserting element cnt =", idx_lst.size());

        pdat.append_subset_to(idx_lst, *this);
    }

} // namespace shamrock::patch
