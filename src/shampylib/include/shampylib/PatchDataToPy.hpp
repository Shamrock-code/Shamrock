// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataToPy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shampylib/PatchDataSetup.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace shamrock {
    template<class T>
    class VecToNumpy;

    template<class T>
    class VecToNumpy {
        public:
        static py::array_t<T> convert(std::vector<T> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i) = vec[i];
            }

            return std::move(ret);
        }
    };

    template<class T>
    class VecToNumpy<sycl::vec<T, 2>> {
        public:
        static py::array_t<T> convert(std::vector<sycl::vec<T, 2>> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len, 2U});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i, 0) = vec[i].x();
                r(i, 1) = vec[i].y();
            }
            return std::move(ret);
        }
    };

    template<class T>
    class VecToNumpy<sycl::vec<T, 3>> {
        public:
        static py::array_t<T> convert(std::vector<sycl::vec<T, 3>> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len, 3U});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i, 0) = vec[i].x();
                r(i, 1) = vec[i].y();
                r(i, 2) = vec[i].z();
            }
            return std::move(ret);
        }
    };

    template<class T>
    class VecToNumpy<sycl::vec<T, 4>> {
        public:
        static py::array_t<T> convert(std::vector<sycl::vec<T, 4>> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len, 4U});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i, 0) = vec[i].x();
                r(i, 1) = vec[i].y();
                r(i, 2) = vec[i].z();
                r(i, 3) = vec[i].w();
            }
            return std::move(ret);
        }
    };

    template<class T>
    class VecToNumpy<sycl::vec<T, 8>> {
        public:
        static py::array_t<T> convert(std::vector<sycl::vec<T, 8>> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len, 8U});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i, 0) = vec[i].s0();
                r(i, 1) = vec[i].s1();
                r(i, 2) = vec[i].s2();
                r(i, 3) = vec[i].s3();
                r(i, 4) = vec[i].s4();
                r(i, 5) = vec[i].s5();
                r(i, 6) = vec[i].s6();
                r(i, 7) = vec[i].s7();
            }
            return std::move(ret);
        }
    };

    template<class T>
    class VecToNumpy<sycl::vec<T, 16>> {
        public:
        static py::array_t<T> convert(std::vector<sycl::vec<T, 16>> vec) {

            u32 len = vec.size();

            py::array_t<T> ret({len, 16U});
            auto r = ret.mutable_unchecked();

            for (u32 i = 0; i < len; i++) {
                r(i, 0)  = vec[i].s0();
                r(i, 1)  = vec[i].s1();
                r(i, 2)  = vec[i].s2();
                r(i, 3)  = vec[i].s3();
                r(i, 4)  = vec[i].s4();
                r(i, 5)  = vec[i].s5();
                r(i, 6)  = vec[i].s6();
                r(i, 7)  = vec[i].s7();
                r(i, 8)  = vec[i].s8();
                r(i, 9)  = vec[i].s9();
                r(i, 10) = vec[i].sA();
                r(i, 11) = vec[i].sB();
                r(i, 12) = vec[i].sC();
                r(i, 13) = vec[i].sD();
                r(i, 14) = vec[i].sE();
                r(i, 15) = vec[i].sF();
            }

            return std::move(ret);
        }
    };

    template<class T>
    void append_to_map(
        std::string key,
        std::vector<std::reference_wrapper<shamrock::patch::PatchDataLayer>> ref_lst,
        py::dict &dic_out) {

        std::vector<T> vec;

        auto appender = [&](auto &field) {
            if (field.get_name() == key) {

                logger::debug_ln("PatchDataToPy", "appending field", key);

                {
                    auto acc = field.get_buf().copy_to_stdvec();
                    u32 len  = field.get_val_cnt();

                    for (u32 i = 0; i < len; i++) {
                        vec.push_back(acc[i]);
                    }
                }
            }
        };

        for (auto &pdat_ref : ref_lst) {
            auto &pdat = pdat_ref.get();
            if (pdat.get_obj_cnt() > 0) {
                pdat.for_each_field<T>([&](auto &field) {
                    appender(field);
                });
            }
        }

        if (!vec.empty()) {
            auto arr = VecToNumpy<T>::convert(vec);

            logger::debug_ln("PatchDataToPy", "adding -> ", key);

            if (dic_out.contains(key.c_str())) {
                throw shambase::make_except_with_loc<std::runtime_error>("the key already exists");
            } else {
                dic_out[key.c_str()] = arr;
            }
        }
    }

    template<class T>
    void append_to_map(
        std::string key,
        std::vector<std::unique_ptr<shamrock::patch::PatchDataLayer>> &lst,
        py::dict &dic_out) {

        std::vector<std::reference_wrapper<shamrock::patch::PatchDataLayer>> ref_lst;
        for (auto &pdat : lst) {
            if (pdat) {
                ref_lst.push_back(*pdat);
            }
        }

        append_to_map<T>(key, ref_lst, dic_out);
    }

    inline py::dict pdat_to_dic(shamrock::patch::PatchDataLayer &pdat) {
        py::dict dic_out;

        std::reference_wrapper<shamrock::patch::PatchDataLayer> ref_pdat = pdat;

        using namespace shamrock;

        for (auto fname : pdat.pdl().get_field_names()) {
            append_to_map<f32>(fname, {ref_pdat}, dic_out);
            append_to_map<f32_2>(fname, {ref_pdat}, dic_out);
            append_to_map<f32_3>(fname, {ref_pdat}, dic_out);
            append_to_map<f32_4>(fname, {ref_pdat}, dic_out);
            append_to_map<f32_8>(fname, {ref_pdat}, dic_out);
            append_to_map<f32_16>(fname, {ref_pdat}, dic_out);
            append_to_map<f64>(fname, {ref_pdat}, dic_out);
            append_to_map<f64_2>(fname, {ref_pdat}, dic_out);
            append_to_map<f64_3>(fname, {ref_pdat}, dic_out);
            append_to_map<f64_4>(fname, {ref_pdat}, dic_out);
            append_to_map<f64_8>(fname, {ref_pdat}, dic_out);
            append_to_map<f64_16>(fname, {ref_pdat}, dic_out);
            append_to_map<u32>(fname, {ref_pdat}, dic_out);
            append_to_map<u64>(fname, {ref_pdat}, dic_out);
            append_to_map<u32_3>(fname, {ref_pdat}, dic_out);
            append_to_map<u64_3>(fname, {ref_pdat}, dic_out);
            append_to_map<i64_3>(fname, {ref_pdat}, dic_out);
        }

        return dic_out;
    }

    template<class T>
    class NumpyToVec;

    template<>
    class NumpyToVec<f64> {
        public:
        static std::vector<f64> convert(py::array_t<f64> arr) {
            if (arr.ndim() != 1) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    shambase::format("expected 1D array for f64 field, got ndim={}", arr.ndim()));
            }

            auto r = arr.unchecked<1>();
            std::vector<f64> vec(static_cast<size_t>(r.shape(0)));
            for (py::ssize_t i = 0; i < r.shape(0); i++) {
                vec[static_cast<size_t>(i)] = r(i);
            }
            return vec;
        }
    };

    template<>
    class NumpyToVec<f64_3> {
        public:
        static std::vector<f64_3> convert(py::array_t<f64> arr) {
            if (arr.ndim() != 2 || arr.shape(1) != 3) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "expected (N, 3) array for f64_3 field");
            }

            auto r = arr.unchecked<2>();
            std::vector<f64_3> vec(static_cast<size_t>(r.shape(0)));
            for (py::ssize_t i = 0; i < r.shape(0); i++) {
                vec[static_cast<size_t>(i)] = f64_3{r(i, 0), r(i, 1), r(i, 2)};
            }
            return vec;
        }
    };

    inline void register_field_io_f64(PatchDataSetup &setup, PatchDataField<f64> &field) {
        std::string name = field.get_name();
        setup.register_getter(name, [&field]() -> py::array_t<f64> {
            return VecToNumpy<f64>::convert(field.get_buf().copy_to_stdvec());
        });
        setup.register_setter(name, [&field](py::array_t<f64> arr) {
            auto vec = NumpyToVec<f64>::convert(arr);
            if (vec.size() != field.get_val_cnt()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "field \"{}\": array size {} does not match field val_cnt {}",
                    field.get_name(),
                    vec.size(),
                    field.get_val_cnt()));
            }
            field.get_buf().copy_from_stdvec(vec);
        });
    }

    inline void register_field_io_f64_3(PatchDataSetup &setup, PatchDataField<f64_3> &field) {
        std::string name = field.get_name();
        setup.register_getter(name, [&field]() -> py::array_t<f64> {
            return VecToNumpy<f64_3>::convert(field.get_buf().copy_to_stdvec());
        });
        setup.register_setter(name, [&field](py::array_t<f64> arr) {
            auto vec = NumpyToVec<f64_3>::convert(arr);
            if (vec.size() != field.get_val_cnt()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "field \"{}\": array size {} does not match field val_cnt {}",
                    field.get_name(),
                    vec.size(),
                    field.get_val_cnt()));
            }
            field.get_buf().copy_from_stdvec(vec);
        });
    }

    inline void register_f64_layout_fields(
        PatchDataSetup &setup, shamrock::patch::PatchDataLayer &pdat) {
        pdat.for_each_field<f64>([&](auto &field) {
            register_field_io_f64(setup, field);
        });
        pdat.for_each_field<f64_3>([&](auto &field) {
            register_field_io_f64_3(setup, field);
        });
    }
} // namespace shamrock
