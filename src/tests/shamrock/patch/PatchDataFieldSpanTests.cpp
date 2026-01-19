// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <experimental/mdspan>
#include <memory>
#include <vector>

TestStart(Unittest, "shamrock/patch/PatchDataFieldSpan", testpatchdatafieldspan, 1) {

    StackEntry stack_loc{};

    using T = f64;

    std::vector<T> test_vals{0, 1, 2, 3, 4, 5, 6, 7};

    u32 cnt_test = test_vals.size();

    {
        // exception for static nvar
        PatchDataField<T> field("test", 1, cnt_test);
        field.override(test_vals, cnt_test);

        auto test = [&]() {
            shamrock::PatchDataFieldSpan<T, 2>(field, 0, cnt_test);
        };

        // cannot bind a PatchDataFieldSpan with static nvar=2 to a PatchDataField with nvar=1
        REQUIRE_EXCEPTION_THROW(test(), std::invalid_argument);
    }

    {
        // nvar 1 case
        PatchDataField<T> field("test", 1, cnt_test);
        field.override(test_vals, cnt_test);

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{span},
            sham::MultiRef{ret},
            cnt_test,
            [](u32 i, auto sp, T *ret_val) {
                ret_val[i] = sp(i, 0);
            });

        REQUIRE_EQUAL_NAMED("dynamic nvar 1| read only", test_vals, ret.copy_to_stdvec());
    }

    {
        // nvar 1 case
        PatchDataField<T> field("test", 1, cnt_test);
        field.override(test_vals, cnt_test);

        shamrock::PatchDataFieldSpan<T, 1> span(field, 0, cnt_test);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{span},
            sham::MultiRef{ret},
            cnt_test,
            [](u32 i, auto sp, T *ret_val) {
                ret_val[i] = sp(i, 0);
            });

        REQUIRE_EQUAL_NAMED("static nvar 1 | read only", test_vals, ret.copy_to_stdvec());
    }

    {
        // nvar 1 case
        PatchDataField<T> field("test", 1, cnt_test);
        field.override(test_vals, cnt_test);

        shamrock::PatchDataFieldSpan<T, 1> span(field, 0, cnt_test);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{span},
            sham::MultiRef{ret},
            cnt_test,
            [](u32 i, auto sp, T *ret_val) {
                ret_val[i] = sp(i);
            });

        REQUIRE_EQUAL_NAMED(
            "static nvar 1 (simple case) | read only", test_vals, ret.copy_to_stdvec());
    }

    {
        // nvar 2 case
        PatchDataField<T> field("test", 2, cnt_test / 2);
        field.override(test_vals, cnt_test);

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test / 2);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{span},
            sham::MultiRef{ret},
            cnt_test / 2,
            [](u32 i, auto sp, T *ret_val) {
                ret_val[i * 2 + 0] = sp(i, 0);
                ret_val[i * 2 + 1] = sp(i, 1);
            });

        REQUIRE_EQUAL_NAMED("dynamic nvar 2| read only", test_vals, ret.copy_to_stdvec());
    }

    {
        // nvar 2 case
        PatchDataField<T> field("test", 2, cnt_test / 2);
        field.override(test_vals, cnt_test);

        shamrock::PatchDataFieldSpan<T, 2> span(field, 0, cnt_test / 2);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{span},
            sham::MultiRef{ret},
            cnt_test / 2,
            [](u32 i, auto sp, T *ret_val) {
                ret_val[i * 2 + 0] = sp(i, 0);
                ret_val[i * 2 + 1] = sp(i, 1);
            });

        REQUIRE_EQUAL_NAMED("static nvar 2 | read only", test_vals, ret.copy_to_stdvec());
    }

    {
        // nvar 1 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 1, cnt_test);

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test,
            [](u32 i, const T *in_val, auto sp) {
                sp(i, 0) = in_val[i];
            });

        REQUIRE_EQUAL_NAMED("dynamic nvar 1| write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 1 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 1, cnt_test);

        shamrock::PatchDataFieldSpan<T, 1> span(field, 0, cnt_test);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test,
            [](u32 i, const T *in_val, auto sp) {
                sp(i, 0) = in_val[i];
            });

        REQUIRE_EQUAL_NAMED("static nvar 1| write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 1 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 1, cnt_test);

        shamrock::PatchDataFieldSpan<T, 1> span(field, 0, cnt_test / 2);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test,
            [](u32 i, const T *in_val, auto sp) {
                sp(i) = in_val[i];
            });

        REQUIRE_EQUAL_NAMED(
            "static nvar 1 (simple case) | write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 2 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 2, cnt_test / 2);

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test / 2);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test / 2,
            [=](u32 i, const T *in_val, auto sp) {
                using nvar_extent = std::extents<u32, std::dynamic_extent, 2>;
                using nvar_mdspan = std::mdspan<const T, nvar_extent, std::layout_right>;
                auto span         = nvar_mdspan(in_val, cnt_test / 2);

                sp(i, 0) = span(i, 0);
                sp(i, 1) = span(i, 1);
            });

        REQUIRE_EQUAL_NAMED("dynamic nvar 2| write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 2 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 2, cnt_test / 2);

        shamrock::PatchDataFieldSpan<T, 2> span(field, 0, cnt_test / 2);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test / 2,
            [=](u32 i, const T *in_val, auto sp) {
                using nvar_extent = std::extents<u32, std::dynamic_extent, 2>;
                using nvar_mdspan = std::mdspan<const T, nvar_extent, std::layout_right>;
                auto span         = nvar_mdspan(in_val, cnt_test / 2);

                sp(i, 0) = span(i, 0);
                sp(i, 1) = span(i, 1);
            });

        REQUIRE_EQUAL_NAMED("static nvar 2| write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 2 case with start offset
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        u32 offset = 5;

        PatchDataField<T> field("test", 2, (cnt_test) / 2 + offset);
        field.field_raz();

        shamrock::PatchDataFieldSpan<T, 2> span(field, offset, cnt_test / 2);

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test / 2,
            [=](u32 i, const T *in_val, auto sp) {
                using nvar_extent = std::extents<u32, std::dynamic_extent, 2>;
                using nvar_mdspan = std::mdspan<const T, nvar_extent, std::layout_right>;
                auto span         = nvar_mdspan(in_val, cnt_test / 2);

                sp(i, 0) = span(i, 0);
                sp(i, 1) = span(i, 1);
            });

        std::vector<T> test_vals_with_offset = std::vector<T>(10, 0);
        test_vals_with_offset.insert(
            test_vals_with_offset.end(), test_vals.begin(), test_vals.end());

        REQUIRE_EQUAL_NAMED(
            "static nvar 2| write + offset", test_vals_with_offset, field.copy_to_stdvec());
    }
}
