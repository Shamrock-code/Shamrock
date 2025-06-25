// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataField.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/container/ResizableBuffer.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <array>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

template<class T>
class PatchDataField {

    ///////////////////////////////////
    // constexpr utilities (using & constexpr vals)
    ///////////////////////////////////

    static constexpr bool isprimitive = std::is_same<T, f32>::value || std::is_same<T, f64>::value
                                        || std::is_same<T, u32>::value
                                        || std::is_same<T, u64>::value;

    // clang-format off
    static constexpr bool is_in_type_list =
        #define X(args) std::is_same<T, args>::value ||
        XMAC_LIST_ENABLED_FIELD false
        #undef X
        ;

    static_assert(
        is_in_type_list,
        "PatchDataField must be one of those types : "

        #define X(args) #args " "
        XMAC_LIST_ENABLED_FIELD
        #undef X
    );
    // clang-format on

    template<bool B, class Tb = void>
    using enable_if_t = typename std::enable_if<B, Tb>;

    using EnableIfPrimitive = enable_if_t<isprimitive>;

    using EnableIfVec = enable_if_t<is_in_type_list && (!isprimitive)>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // member fields
    ////////////////////////////////////////////////////////////////////////////////////////////////

    sham::DeviceBuffer<T> buf;

    std::string field_name;

    u32 nvar;    // number of variable per object
    u32 obj_cnt; // number of contained object

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////////////////

    public:
    inline PatchDataField(PatchDataField &&other) noexcept
        : buf(std::move(other.buf)), field_name(std::move(other.field_name)),
          nvar(std::move(other.nvar)), obj_cnt(std::move(other.obj_cnt)) {} // move constructor

    inline PatchDataField &operator=(PatchDataField &&other) noexcept {
        buf        = std::move(other.buf);
        field_name = std::move(other.field_name);
        nvar       = std::move(other.nvar);
        obj_cnt    = std::move(other.obj_cnt);

        return *this;
    } // move assignment

    // TODO find a way to add particles easily cf setup require public vector

    using Field_type = T;

    inline PatchDataField(std::string name, u32 nvar)
        : field_name(std::move(name)), nvar(nvar), obj_cnt(0),
          buf(0, shamsys::instance::get_compute_scheduler_ptr()) {};

    inline PatchDataField(std::string name, u32 nvar, u32 obj_cnt)
        : field_name(std::move(name)), nvar(nvar), obj_cnt(obj_cnt),
          buf(obj_cnt * nvar, shamsys::instance::get_compute_scheduler_ptr()) {};

    inline PatchDataField(const PatchDataField &other)
        : field_name(other.field_name), nvar(other.nvar), obj_cnt(other.obj_cnt),
          buf(other.buf.copy()) {}

    inline PatchDataField(
        sham::DeviceBuffer<T> &&moved_buf, u32 obj_cnt, std::string name, u32 nvar)
        : obj_cnt(obj_cnt), field_name(name), nvar(nvar),
          buf(std::forward<sham::DeviceBuffer<T>>(moved_buf)) {}

    inline PatchDataField(sycl::buffer<T> &&moved_buf, u32 obj_cnt, std::string name, u32 nvar)
        : obj_cnt(obj_cnt), field_name(name), nvar(nvar),
          buf(std::forward<sycl::buffer<T>>(moved_buf),
              obj_cnt * nvar,
              shamsys::instance::get_compute_scheduler_ptr()) {}

    PatchDataField &operator=(const PatchDataField &other) = delete;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // member functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline PatchDataField duplicate() const {
        const PatchDataField &current = *this;
        return PatchDataField(current);
    }

    inline PatchDataField duplicate(std::string new_name) const {
        const PatchDataField &current = *this;
        PatchDataField ret            = PatchDataField(current);
        ret.field_name                = new_name;
        return ret;
    }

    inline std::unique_ptr<PatchDataField> duplicate_to_ptr() const {
        const PatchDataField &current = *this;
        return std::make_unique<PatchDataField>(current);
    }

    inline sham::DeviceBuffer<T> &get_buf() { return buf; }
    inline const sham::DeviceBuffer<T> &get_buf() const { return buf; }

    [[nodiscard]] inline bool is_empty() const { return get_obj_cnt() == 0; }

    [[nodiscard]] inline u64 memsize() const { return buf.get_mem_usage(); }

    [[nodiscard]] inline const u32 &get_nvar() const { return nvar; }

    [[nodiscard]] inline const u32 &get_obj_cnt() const { return obj_cnt; }

    /**
     * @brief Get the number of values stored in the field.
     *
     * This function was introduced to replace the legacy one size() which could be confused with
     * the of the buffer, which is not required to be the same.
     *
     * @return u32 the total number of values of the field, which is the product of the number of
     * objects and the number of variables per object.
     */
    [[nodiscard]] inline u32 get_val_cnt() const { return get_obj_cnt() * get_nvar(); }

    [[nodiscard]] inline const std::string &get_name() const { return field_name; }

    // TODO add overflow check
    void resize(u32 new_obj_cnt);

    void reserve(u32 new_obj_cnt);

    void expand(u32 obj_to_add);

    void shrink(u32 obj_to_rem);

    void insert_element(T v);

    void apply_offset(T off);

    void insert(PatchDataField<T> &f2);

    void overwrite(PatchDataField<T> &f2, u32 obj_cnt);

    void override(sycl::buffer<T> &data, u32 cnt);

    void override(std::vector<T> &data, u32 cnt);

    void override(const T val);

    inline void synchronize_buf() { buf.synchronize(); }

    inline std::vector<T> copy_to_stdvec() {
        auto tmp = buf.copy_to_stdvec();
        tmp.resize(get_val_cnt());
        return tmp;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Span utilities
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Returns a shamrock::PatchDataFieldSpan pointing to the current PatchDataField.
     *
     * @tparam nvar The number of variables in the span.
     * @return A shamrock::PatchDataFieldSpan pointing to this PatchDataField.
     */
    template<u32 nvar, bool is_pointer_access = shamrock::access_t_span>
    inline shamrock::PatchDataFieldSpan<T, nvar, is_pointer_access> get_span() {
        StackEntry stack_loc{};
        return shamrock::PatchDataFieldSpan<T, nvar, is_pointer_access>(*this, 0, get_obj_cnt());
    }

    /**
     * @brief Returns a shamrock::PatchDataFieldSpan pointing to the current PatchDataField.
     *
     * @return A shamrock::PatchDataFieldSpan pointing to this PatchDataField with dynamic number
     * of variables.
     */
    inline shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar> get_span_nvar_dynamic() {
        StackEntry stack_loc{};
        return get_span<shamrock::dynamic_nvar, shamrock::access_t_span>();
    }

    inline shamrock::PatchDataFieldSpan<T, shamrock::dynamic_nvar, shamrock::access_t_pointer>
    get_pointer_span() {
        StackEntry stack_loc{};
        return get_span<shamrock::dynamic_nvar, shamrock::access_t_pointer>();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // get_subsets utilities
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Get the ids set where object
     *
     * @tparam Lambdacd
     * @tparam Args
     * @param cd_true
     * @param args
     * @return std::set<u32>
     */
    template<class Lambdacd, class... Args>
    inline std::set<u32> get_ids_set_where(Lambdacd &&cd_true, Args... args) {
        StackEntry stack_loc{};
        std::set<u32> idx_cd{};
        if (get_obj_cnt() > 0) {
            auto acc = get_buf().copy_to_stdvec();

            for (u32 i = 0; i < get_obj_cnt(); i++) {
                if (cd_true(acc, i * nvar, args...)) {
                    idx_cd.insert(i);
                }
            }
        }
        return idx_cd;
    }

    /**
     * @brief Same function as @see PatchDataField#get_ids_set_where but return a std::vector of the
     * found index
     *
     * @tparam Lambdacd
     * @tparam Args
     * @param cd_true
     * @param args
     * @return std::vector<u32>
     */
    template<class Lambdacd, class... Args>
    inline std::vector<u32> get_ids_vec_where(Lambdacd &&cd_true, Args... args) {
        StackEntry stack_loc{};
        std::vector<u32> idx_cd{};
        if (get_obj_cnt() > 0) {
            auto acc = buf.copy_to_stdvec();

            for (u32 i = 0; i < get_obj_cnt(); i++) {
                if (cd_true(acc, i * nvar, args...)) {
                    idx_cd.push_back(i);
                }
            }
        }
        return idx_cd;
    }

    /**
     * @brief Same function as @see PatchDataField#get_ids_set_where but return a optional
     * sycl::buffer of the found index
     *
     * @tparam Lambdacd
     * @tparam Args
     * @param cd_true
     * @param args
     * @return std::vector<u32>
     */
    template<class Lambdacd, class... Args>
    inline std::tuple<std::optional<sycl::buffer<u32>>, u32>
    get_ids_buf_where(Lambdacd &&cd_true, Args... args) {
        StackEntry stack_loc{};

        if (get_obj_cnt() > 0) {

            // buffer of booleans to store result of the condition
            sycl::buffer<u32> mask(get_obj_cnt());

            sham::EventList depends_list;
            const T *acc = buf.get_read_access(depends_list);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            auto e = q.submit(depends_list, [&, args...](sycl::handler &cgh) {
                sycl::accessor acc_mask{mask, cgh, sycl::write_only, sycl::no_init};
                u32 nvar_field = nvar;

                shambase::parralel_for(
                    cgh, get_obj_cnt(), "PatchdataField::get_ids_buf_where", [=](u32 id) {
                        acc_mask[id] = cd_true(acc, id * nvar_field, args...);
                    });
            });

            buf.complete_event_state(e);

            return shamalgs::numeric::stream_compact(
                shamsys::instance::get_compute_queue(), mask, get_obj_cnt());
        } else {
            return {std::nullopt, 0};
        }
    }

    template<class Lambdacd>
    [[deprecated("please use one of the PatchDataField::get_ids_..._where functions instead")]]
    std::vector<u32> get_elements_with_range(Lambdacd &&cd_true, T vmin, T vmax);

    /**
     * @brief Get the indicies of the elements in half open interval
     *
     * @tparam LambdaCd
     * @param vmin
     * @param vmax
     * @return std::tuple<std::optional<sycl::buffer<u32>>, u32>
     */
    template<class LambdaCd>
    [[deprecated("please use one of the PatchDataField::get_ids_..._where functions instead")]]
    std::tuple<std::optional<sycl::buffer<u32>>, u32> get_elements_in_half_open(T vmin, T vmax);

    template<class Lambdacd>
    [[deprecated("please use one of the PatchDataField::get_ids_..._where functions instead")]]
    std::unique_ptr<sycl::buffer<u32>>
    get_elements_with_range_buf(Lambdacd &&cd_true, T vmin, T vmax);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Lambdacd>
    void check_err_range(Lambdacd &&cd_true, T vmin, T vmax, std::string add_log = "");

    void extract_element(u32 pidx, PatchDataField<T> &to);

    bool check_field_match(PatchDataField<T> &f2);

    inline void field_raz() {
        logger::debug_ln("PatchDataField", "raz : ", field_name);
        override(shambase::VectorProperties<T>::get_zero());
    }

    /**
     * @brief Copy all objects in idxs to pfield
     *
     * @param idxs
     * @param pfield
     */
    void append_subset_to(const std::vector<u32> &idxs, PatchDataField &pfield);
    void append_subset_to(sycl::buffer<u32> &idxs_buf, u32 sz, PatchDataField &pfield);

    inline PatchDataField make_new_from_subset(sycl::buffer<u32> &idxs_buf, u32 sz) {
        PatchDataField pfield(field_name, nvar);
        append_subset_to(idxs_buf, sz, pfield);
        return pfield;
    }

    void gen_mock_data(u32 obj_cnt, std::mt19937 &eng);

    /**
     * @brief this function remaps the patchdatafield like so
     *   val[id] = val[index_map[id]]
     *   index map describe : at index i, we will have the value that was at index_map[i]
     *
     * This function can be used to apply the result of a sort to the field
     *
     * @param index_map
     * @param len the length of the map (must match with the current count)
     */
    void index_remap(sham::DeviceBuffer<u32> &index_map, u32 len);

    /**
     * @brief this function remaps the patchdatafield like so
     *   val[id] = val[index_map[id]]
     *   index map describe : at index i, we will have the value that was at index_map[i]
     * This function will resize the current field to the specified length
     *
     * This function can be used to apply the result of a sort to the field
     *
     * @param index_map
     * @param len the length of the map
     */
    void index_remap_resize(sham::DeviceBuffer<u32> &index_map, u32 len);

    /**
     * @brief minimal serialization
     * assuming the user know the layout of the field
     *
     * @param serializer
     */
    void serialize_buf(shamalgs::SerializeHelper &serializer);

    /**
     * @brief deserialize a field inverse of serialize_buf
     *
     * @param serializer
     * @param field_name
     * @param nvar
     * @return PatchDataField
     */
    static PatchDataField
    deserialize_buf(shamalgs::SerializeHelper &serializer, std::string field_name, u32 nvar);

    /**
     * @brief record the size usage of the serialization using serialize_buf
     *
     * @return u64
     */
    shamalgs::SerializeSize serialize_buf_byte_size();

    /**
     * @brief serialize everything in the class
     *
     * @param serializer
     */
    void serialize_full(shamalgs::SerializeHelper &serializer);

    /**
     * @brief deserialize a field inverse of serialize_full
     *
     * @param serializer
     * @return PatchDataField
     */
    static PatchDataField deserialize_full(shamalgs::SerializeHelper &serializer);

    /**
     * @brief give the size usage of serialize_full
     *
     * @return u64
     */
    shamalgs::SerializeSize serialize_full_byte_size();

    T compute_max();
    T compute_min();
    T compute_sum();

    shambase::VecComponent<T> compute_dot_sum();

    bool has_nan();
    bool has_inf();
    bool has_nan_or_inf();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // static member functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    static PatchDataField<T> mock_field(u64 seed, u32 obj_cnt, std::string name, u32 nvar);
    static PatchDataField<T>
    mock_field(u64 seed, u32 obj_cnt, std::string name, u32 nvar, T vmin, T vmax);
};

// TODO add overflow check
template<class T>
inline void PatchDataField<T>::resize(u32 new_obj_cnt) {

    u32 new_size = new_obj_cnt * nvar;
    // field_data.resize(new_size);

    buf.resize(new_size);

    obj_cnt = new_obj_cnt;
}

template<class T>
inline void PatchDataField<T>::reserve(u32 new_obj_cnt) {

    u32 add_cnt = new_obj_cnt * nvar;
    buf.reserve(add_cnt);
}

template<class T>
inline void PatchDataField<T>::expand(u32 obj_to_add) {
    resize(obj_cnt + obj_to_add);
}

template<class T>
inline void PatchDataField<T>::shrink(u32 obj_to_rem) {

    if (obj_to_rem > obj_cnt) {

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "impossible to remove more object than there is in the patchdata field");
    }

    resize(obj_cnt - obj_to_rem);
}

template<class T>
inline void PatchDataField<T>::overwrite(PatchDataField<T> &f2, u32 obj_cnt) {
    StackEntry stack_loc{};
    buf.copy_from(f2.buf, obj_cnt * f2.nvar);
}

template<class T>
inline void PatchDataField<T>::override(sycl::buffer<T> &data, u32 cnt) {
    StackEntry stack_loc{};
    buf.copy_from_sycl_buffer(data, cnt);
}

template<class T>
inline void PatchDataField<T>::override(std::vector<T> &data, u32 cnt) {
    StackEntry stack_loc{};
    buf.copy_from_stdvec(data, cnt);
}

template<class T>
inline void PatchDataField<T>::override(const T val) {
    StackEntry stack_loc{};
    buf.fill(val);
}

template<class T>
template<class Lambdacd>
inline std::vector<u32>
PatchDataField<T>::get_elements_with_range(Lambdacd &&cd_true, T vmin, T vmax) {
    StackEntry stack_loc{};
    std::vector<u32> idxs;

    /* Possible GPU version
    sycl::buffer<u32> valid {size()};

    shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
        sycl::accessor acc {shambase::get_check_ref(get_buf()), cgh, sycl::read_only};
        sycl::accessor bools {valid, cgh,sycl::write_only,sycl::no_init};

        shambase::parralel_for(cgh,size(),"get_element_with_range",[=](u32 i){
            bools[i] = (cd_true(acc[i], vmin, vmax)) ? 1 : 0;
        });

    });

    std::tuple<std::optional<sycl::buffer<u32>>, u32> ret =
        shamalgs::numeric::stream_compact(shamsys::instance::get_compute_queue(), valid, size());

    std::vector<u32> idxs;

    {
        if(std::get<0>(ret).has_value()){
            idxs = shamalgs::memory::buf_to_vec(*std::get<0>(ret), std::get<1>(ret));
        }
    }
    */

    if (nvar != 1) {
        shambase::throw_unimplemented();
    }

    {
        auto acc = buf.copy_to_stdvec();

        for (u32 i = 0; i < get_val_cnt(); i++) {
            if (cd_true(acc[i], vmin, vmax)) {
                idxs.push_back(i);
            }
        }
    }

    return idxs;
}

template<class T>
template<class Lambdacd>
inline std::unique_ptr<sycl::buffer<u32>>
PatchDataField<T>::get_elements_with_range_buf(Lambdacd &&cd_true, T vmin, T vmax) {
    std::vector<u32> idxs = get_elements_with_range(std::forward<Lambdacd>(cd_true), vmin, vmax);
    if (idxs.empty()) {
        return {};
    } else {
        return std::make_unique<sycl::buffer<u32>>(shamalgs::memory::vec_to_buf(idxs));
    }
}

class PatchDataRangeCheckError : public std::exception {
    public:
    explicit PatchDataRangeCheckError(const char *message) : msg_(message) {}

    explicit PatchDataRangeCheckError(const std::string &message) : msg_(message) {}

    ~PatchDataRangeCheckError() noexcept override = default;

    [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

    protected:
    std::string msg_;
};

template<class T>
template<class Lambdacd>
inline void
PatchDataField<T>::check_err_range(Lambdacd &&cd_true, T vmin, T vmax, std::string add_log) {
    StackEntry stack_loc{};

    if (is_empty()) {
        return;
    }

    if (nvar != 1) {
        shambase::throw_unimplemented();
    }

    bool error = false;
    {
        auto acc    = buf.copy_to_stdvec();
        u32 err_cnt = 0;

        for (u32 i = 0; i < get_val_cnt(); i++) {
            if (!cd_true(acc[i], vmin, vmax)) {
                logger::err_ln(
                    "PatchDataField",
                    "obj =",
                    i,
                    "->",
                    acc[i],
                    "not in range [",
                    vmin,
                    ",",
                    vmax,
                    "]");
                error = true;
                err_cnt++;
                if (err_cnt > 50) {
                    logger::err_ln("PatchDataField", "...");
                    break;
                }
            }
        }
    }

    if (error) {
        logger::err_ln("PatchDataField", "additional infos :", add_log);
        throw shambase::make_except_with_loc<PatchDataRangeCheckError>("obj not in range");
    }
}
