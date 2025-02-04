// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

TestStart(Unittest, "shamrock/patch/PatchDataFieldSpan", testpatchdatafieldspan, 1) {

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

        sham::EventList depends_list;

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

        sham::EventList depends_list;

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

        sham::EventList depends_list;

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

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::EventList depends_list;

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

        shamrock::PatchDataFieldSpan<T, 2> span(field, 0, cnt_test);

        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());

        sham::EventList depends_list;

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

        sham::EventList depends_list;

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

        sham::EventList depends_list;

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

        shamrock::PatchDataFieldSpan<T, 1> span(field, 0, cnt_test);

        sham::EventList depends_list;

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

        shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> span(field, 0, cnt_test);

        sham::EventList depends_list;

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test / 2,
            [](u32 i, const T *in_val, auto sp) {
                sp(i, 0) = in_val[i * 2 + 0];
                sp(i, 1) = in_val[i * 2 + 1];
            });

        REQUIRE_EQUAL_NAMED("dynamic nvar 2| write", test_vals, field.copy_to_stdvec());
    }

    {
        // nvar 2 case
        sham::DeviceBuffer<T> ret(test_vals.size(), shamsys::instance::get_compute_scheduler_ptr());
        ret.copy_from_stdvec(test_vals);

        PatchDataField<T> field("test", 2, cnt_test / 2);

        shamrock::PatchDataFieldSpan<T, 2> span(field, 0, cnt_test);

        sham::EventList depends_list;

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            sham::MultiRef{ret},
            sham::MultiRef{span},
            cnt_test / 2,
            [](u32 i, const T *in_val, auto sp) {
                sp(i, 0) = in_val[i * 2 + 0];
                sp(i, 1) = in_val[i * 2 + 1];
            });

        REQUIRE_EQUAL_NAMED("static nvar 2| write", test_vals, field.copy_to_stdvec());
    }
}
