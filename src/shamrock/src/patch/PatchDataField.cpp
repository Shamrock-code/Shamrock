// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchDataField.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/algorithm.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <memory>

template<class T>
class Kernel_Extract_element;

template<class T>
void PatchDataField<T>::extract_element(u32 pidx, PatchDataField<T> &to) {

    auto fast_extract_ptr = [](u32 idx, u32 length, auto cnt) {
        T end_ = cnt[length - 1];
        T extr = cnt[idx];

        cnt[idx] = end_;

        return extr;
    };

    auto sub_extract
        = [fast_extract_ptr](u32 pidx, PatchDataField<T> &from, PatchDataField<T> &to) {
              const u32 nvar        = from.get_nvar();
              const u32 idx_val     = pidx * nvar;
              const u32 idx_out_val = to.get_val_cnt();

              u32 from_sz = from.get_val_cnt();

              to.expand(1);

              {

                  auto &buf_to   = to.get_buf();
                  auto &buf_from = from.get_buf();

                  sham::EventList depends_list;
                  T *acc_to   = buf_to.get_write_access(depends_list);
                  T *acc_from = buf_from.get_write_access(depends_list);

                  sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                  auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                      const u32 nvar_loc = nvar;

                      cgh.single_task<Kernel_Extract_element<T>>([=]() {
                          for (u32 i = nvar_loc - 1; i < nvar_loc; i--) {
                              acc_to[idx_out_val + i]
                                  = (fast_extract_ptr(idx_val + i, from_sz, acc_from));
                          }
                      });
                  });

                  buf_to.complete_event_state(e);
                  buf_from.complete_event_state(e);
              }

              from.shrink(1);
          };

    sub_extract(pidx, *this, to);
}

template<class T>
bool PatchDataField<T>::check_field_match(PatchDataField<T> &f2) {
    bool match = true;

    match = match && (field_name == f2.field_name);
    match = match && (nvar == f2.nvar);
    match = match && (obj_cnt == f2.obj_cnt);

    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
    match     = match && shamalgs::equals(sptr, buf, f2.buf, obj_cnt * nvar);

    return match;
}

template<class T>
class PdatField_append_subset_to;

template<class T>
void PatchDataField<T>::append_subset_to(
    sycl::buffer<u32> &idxs_buf, u32 sz, PatchDataField &pfield) {

    if (pfield.nvar != nvar)
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "field must be similar for extraction");

    const u32 start_enque = pfield.get_val_cnt();

    const u32 nvar = get_nvar();

    pfield.expand(sz);

    auto &buf_other = pfield.get_buf();

#ifdef false
    {
        sycl::host_accessor acc_idxs{idxs_buf, sycl::read_only};
        sycl::host_accessor acc_curr{*buf, sycl::read_only};
        sycl::host_accessor acc_other{*buf_other, sycl::write_only, sycl::no_init};

        const u32 nvar_loc        = nvar;
        const u32 start_enque_loc = start_enque;

        for (u32 gid = 0; gid < sz; gid++) {

            u32 _sz = sz;

            const u32 idx_extr = acc_idxs[gid] * nvar_loc;
            const u32 idx_push = start_enque_loc + gid * nvar_loc;

            for (u32 a = 0; a < nvar_loc; a++) {
                auto tmp                = acc_curr[idx_extr + a];
                acc_other[idx_push + a] = tmp;
            }
        }

        // printf("extracting : (%u)\n",sz);
        // for(u32 i = 0; i < sz; i++){
        //     printf("%u ",acc_idxs[i]);
        // }
        // printf("\n");
    }
#endif

    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
    auto &q   = sptr->get_queue();

    sham::EventList depends_list;

    const T *acc_curr = buf.get_read_access(depends_list);
    T *acc_other      = buf_other.get_write_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        sycl::accessor acc_idxs{idxs_buf, cgh, sycl::read_only};

        const u32 nvar_loc        = nvar;
        const u32 start_enque_loc = start_enque;

        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> i) {
            const u32 gid = i.get_linear_id();

            const u32 idx_extr = acc_idxs[gid] * nvar_loc;
            const u32 idx_push = start_enque_loc + gid * nvar_loc;

            for (u32 a = 0; a < nvar_loc; a++) {
                acc_other[idx_push + a] = acc_curr[idx_extr + a];
            }
        });
    });

    buf.complete_event_state(e);
    buf_other.complete_event_state(e);
}

template<class T>
void PatchDataField<T>::append_subset_to(const std::vector<u32> &idxs, PatchDataField &pfield) {

    if (pfield.nvar != nvar) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "field must be similar for extraction");
    }

    u32 sz = idxs.size();

    if (sz > 0) {
        sycl::buffer<u32> idxs_buf(idxs.data(), sz);
        append_subset_to(idxs_buf, sz, pfield);
    }
}

template<class T>
class PdatField_insert_element;

template<class T>
void PatchDataField<T>::insert_element(T v) {
    if (nvar != 1) {
        shambase::throw_unimplemented();
    }
    u32 ins_pos = get_val_cnt();
    expand(1);

    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
    auto &q   = sptr->get_queue();

    sham::EventList depends_list;
    T *acc = get_buf().get_write_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        auto id_ins = ins_pos;
        auto val    = v;

        cgh.single_task<PdatField_insert_element<T>>([=]() {
            acc[id_ins] = val;
        });
    });

    get_buf().complete_event_state(e);
}

template<class T>
class PdatField_apply_offset;

template<class T>
void PatchDataField<T>::apply_offset(T off) {

    if (get_obj_cnt() > 0) {

        auto sptr = shamsys::instance::get_compute_scheduler_ptr();
        auto &q   = sptr->get_queue();

        sham::EventList depends_list;
        T *acc = get_buf().get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            auto val = off;

            cgh.parallel_for<PdatField_apply_offset<T>>(
                sycl::range<1>{get_val_cnt()}, [=](sycl::id<1> idx) {
                    acc[idx] += val;
                });
        });
        get_buf().complete_event_state(e);
    }
}

template<class T>
class PdatField_insert;

template<class T>
void PatchDataField<T>::insert(PatchDataField<T> &f2) {

    u32 f2_len = f2.get_obj_cnt();

    if (f2_len > 0) {
        logger::debug_sycl_ln("PatchDataField", "expand field buf by N =", f2_len);

        const u32 old_val_cnt = get_val_cnt(); // field_data.size();
        expand(f2.obj_cnt);

        auto sptr = shamsys::instance::get_compute_scheduler_ptr();
        auto &q   = sptr->get_queue();

        sham::EventList depends_list;
        T *acc          = get_buf().get_write_access(depends_list);
        const T *acc_f2 = f2.get_buf().get_read_access(depends_list);

        logger::debug_sycl_ln("PatchDataField", "write values");
        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            const u32 idx_st = old_val_cnt;

            cgh.parallel_for<PdatField_insert<T>>(
                sycl::range<1>{f2.get_val_cnt()}, [=](sycl::id<1> idx) {
                    acc[idx_st + idx] = acc_f2[idx];
                });
        });

        get_buf().complete_event_state(e);
        f2.get_buf().complete_event_state(e);

    } else {
        logger::debug_sycl_ln("PatchDataField", "expand field buf (skip f2 is empty)");
    }
}

template<class T>
void PatchDataField<T>::index_remap_resize(sham::DeviceBuffer<u32> &index_map, u32 len) {

    if (!buf.is_empty()) {

        auto sched_ptr = shamsys::instance::get_compute_scheduler_ptr();

        auto get_new_buf = [&]() {
            if (nvar == 1) {
                return shamalgs::algorithm::index_remap(sched_ptr, buf, index_map, len);
            } else {
                return shamalgs::algorithm::index_remap_nvar(sched_ptr, buf, index_map, len, nvar);
            }
        };

        buf = get_new_buf();
    }

    obj_cnt = len;
}

template<class T>
void PatchDataField<T>::index_remap(sham::DeviceBuffer<u32> &index_map, u32 len) {

    if (len != get_obj_cnt()) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "the match of the new index map does not match with the patchdatafield obj count");
    }

    index_remap_resize(index_map, len);
}

template<class T>
PatchDataField<T>
PatchDataField<T>::mock_field(u64 seed, u32 obj_cnt, std::string name, u32 nvar, T vmin, T vmax) {

    std::vector<T> buf = shamalgs::random::mock_vector<T>(seed, obj_cnt * nvar, vmin, vmax);
    PatchDataField<T> ret(name, nvar, obj_cnt);
    ret.get_buf().copy_from_stdvec(buf);

    return ret;
}

template<class T>
PatchDataField<T> PatchDataField<T>::mock_field(u64 seed, u32 obj_cnt, std::string name, u32 nvar) {
    using Prop = shambase::VectorProperties<T>;
    return PatchDataField<T>::mock_field(
        seed, obj_cnt, name, nvar, Prop::get_min(), Prop::get_max());
}

template<class T>
void PatchDataField<T>::serialize_buf(shamalgs::SerializeHelper &serializer) {
    StackEntry stack_loc{false};
    serializer.write(obj_cnt);
    logger::debug_sycl_ln("PatchDataField", "serialize patchdatafield len=", obj_cnt);
    if (obj_cnt > 0) {
        serializer.write_buf(buf, obj_cnt * nvar);
    }
}

template<class T>
PatchDataField<T> PatchDataField<T>::deserialize_buf(
    shamalgs::SerializeHelper &serializer, std::string field_name, u32 nvar) {
    StackEntry stack_loc{false};
    u32 cnt;
    serializer.load(cnt);
    logger::debug_sycl_ln("PatchDataField", "deserialize patchdatafield len=", cnt);

    if (cnt > 0) {
        sham::DeviceBuffer<T> buf(cnt * nvar, serializer.get_device_scheduler());
        serializer.load_buf(buf, cnt * nvar);
        return PatchDataField<T>(std::move(buf), cnt, field_name, nvar);
    } else {
        return PatchDataField<T>(field_name, nvar, cnt);
    }
}

template<class T>
shamalgs::SerializeSize PatchDataField<T>::serialize_buf_byte_size() {

    using H = shamalgs::SerializeHelper;
    return H::serialize_byte_size<u32>() + H::serialize_byte_size<T>(obj_cnt * nvar);
}

template<class T>
void PatchDataField<T>::serialize_full(shamalgs::SerializeHelper &serializer) {
    StackEntry stack_loc{false};
    serializer.write(nvar);
    serializer.write(field_name);
    serialize_buf(serializer);
}

template<class T>
shamalgs::SerializeSize PatchDataField<T>::serialize_full_byte_size() {
    using H = shamalgs::SerializeHelper;
    return (H::serialize_byte_size<u32>()) + H::serialize_byte_size(field_name)
           + serialize_buf_byte_size();
}

template<class T>
PatchDataField<T> PatchDataField<T>::deserialize_full(shamalgs::SerializeHelper &serializer) {
    StackEntry stack_loc{false};
    u32 nvar;
    serializer.load(nvar);
    std::string field_name;
    serializer.load(field_name);

    return deserialize_buf(serializer, field_name, nvar);
}

template<class T>
T PatchDataField<T>::compute_max() {
    StackEntry stack_loc{};
    if (is_empty()) {
        throw shambase::make_except_with_loc<std::invalid_argument>("the field is empty");
    }

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::max(shamsys::instance::get_compute_queue(), tmp, 0, obj_cnt * nvar);
}

template<class T>
T PatchDataField<T>::compute_min() {
    StackEntry stack_loc{};
    if (is_empty()) {
        throw shambase::make_except_with_loc<std::invalid_argument>("the field is empty");
    }

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::min(shamsys::instance::get_compute_queue(), tmp, 0, obj_cnt * nvar);
}

template<class T>
T PatchDataField<T>::compute_sum() {
    StackEntry stack_loc{};
    if (is_empty()) {
        throw shambase::make_except_with_loc<std::invalid_argument>("the field is empty");
    }

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::sum(shamsys::instance::get_compute_queue(), tmp, 0, obj_cnt * nvar);
}

template<class T>
shambase::VecComponent<T> PatchDataField<T>::compute_dot_sum() {
    StackEntry stack_loc{};
    if (is_empty()) {
        throw shambase::make_except_with_loc<std::invalid_argument>("the field is empty");
    }

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::dot_sum(
        shamsys::instance::get_compute_queue(), tmp, 0, obj_cnt * nvar);
}

template<class T>
bool PatchDataField<T>::has_nan() {
    StackEntry stack_loc{};

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::has_nan(shamsys::instance::get_compute_queue(), tmp, get_val_cnt());
}
template<class T>
bool PatchDataField<T>::has_inf() {
    StackEntry stack_loc{};

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::has_inf(shamsys::instance::get_compute_queue(), tmp, get_val_cnt());
}
template<class T>
bool PatchDataField<T>::has_nan_or_inf() {
    StackEntry stack_loc{};

    auto tmp = buf.copy_to_sycl_buffer();

    return shamalgs::reduction::has_nan_or_inf(
        shamsys::instance::get_compute_queue(), tmp, get_val_cnt());
}

//////////////////////////////////////////////////////////////////////////
// Define the patchdata field for all classes in XMAC_LIST_ENABLED_FIELD
//////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN
    #define X(a) template class PatchDataField<a>;
XMAC_LIST_ENABLED_FIELD
    #undef X
#endif
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// data mocking for patchdata field
//////////////////////////////////////////////////////////////////////////

const u32 obj_mock_cnt = 6000;

#ifndef DOXYGEN
template<>
void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);

    std::vector<f32> out(obj_cnt * nvar);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32(distf64(eng));
    }

    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f32_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f32_2> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32_2{distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f32_3> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32_3{distf64(eng), distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f32_4> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f32_8> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32_8{
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f32_16> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f32_16{
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64(distf64(eng));
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64_2> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64_2{distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64_3> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64_3{distf64(eng), distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64_4> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64_8> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64_8{
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    std::vector<f64_16> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = f64_16{
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng),
            distf64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1, obj_mock_cnt);

    std::vector<u32> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = distu32(eng);
    }
    buf.copy_from_stdvec(out);
}
template<>
void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1, obj_mock_cnt);

    std::vector<u64> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = distu64(eng);
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<u32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1, obj_mock_cnt);

    std::vector<u32_3> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = u32_3{distu32(eng), distu32(eng), distu32(eng)};
    }
    buf.copy_from_stdvec(out);
}
template<>
void PatchDataField<u64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1, obj_mock_cnt);

    std::vector<u64_3> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = u64_3{distu64(eng), distu64(eng), distu64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<i64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<i64> disti64(1, obj_mock_cnt);

    std::vector<i64_3> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = i64_3{disti64(eng), disti64(eng), disti64(eng)};
    }
    buf.copy_from_stdvec(out);
}

template<>
void PatchDataField<i64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<i64> disti64(1, obj_mock_cnt);

    std::vector<i64> out(obj_cnt * nvar);

    for (u32 i = 0; i < get_val_cnt(); i++) {
        out[i] = i64{disti64(eng)};
    }
    buf.copy_from_stdvec(out);
}
#endif
