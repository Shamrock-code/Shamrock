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
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <array>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>

// TODO find a way to add particles easily cf setup require public vector

template<class T>
class PatchDataField {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // constexpr utilities (using & constexpr vals)
    ////////////////////////////////////////////////////////////////////////////////////////////////

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

    ////////////////////////////////////////////////////////////////////////////////////////////////

    static constexpr bool isprimitive = std::is_same<T, f32>::value || std::is_same<T, f64>::value
                                        || std::is_same<T, u32>::value
                                        || std::is_same<T, u64>::value;

    template<bool B, class Tb = void>
    using enable_if_t = typename std::enable_if<B, Tb>;

    using EnableIfPrimitive = enable_if_t<isprimitive>;

    using EnableIfVec = enable_if_t<is_in_type_list && (!isprimitive)>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // constexpr utilities (using & constexpr vals) (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Member fields
    ////////////////////////////////////////////////////////////////////////////////////////////////

    sham::DeviceBuffer<T> buf; ///< the buffer storing the data

    std::string field_name; ///< the name of the field

    u32 nvar;    ///< number of variable per object
    u32 obj_cnt; ///< number of contained object

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Member fields (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////

    public:
    /// The type of the field
    using Field_type = T;

    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Copy & move constructors

    /**
     * @brief Move constructor for PatchDataField.
     *
     * @param other The PatchDataField object to move from.
     */
    inline PatchDataField(PatchDataField &&other) noexcept
        : buf(std::move(other.buf)), field_name(std::move(other.field_name)),
          nvar(std::move(other.nvar)), obj_cnt(std::move(other.obj_cnt)) {}

    /**
     * @brief Move assignment operator for PatchDataField.
     *
     * @param other The PatchDataField object to move from.
     * @return A reference to the moved-to PatchDataField object.
     */
    inline PatchDataField &operator=(PatchDataField &&other) noexcept {
        buf        = std::move(other.buf);
        field_name = std::move(other.field_name);
        nvar       = std::move(other.nvar);
        obj_cnt    = std::move(other.obj_cnt);

        return *this;
    }

    /// Delete implicit copy constructor
    PatchDataField &operator=(const PatchDataField &other) = delete;

    /// Copy constructor for PatchDataField
    inline PatchDataField(const PatchDataField &other)
        : field_name(other.field_name), nvar(other.nvar), obj_cnt(other.obj_cnt),
          buf(other.buf.copy()) {}

    // Generic constructors

    /**
     * @brief Construct a new PatchDataField object with empty buffer.
     *
     * @param name The name of the field.
     * @param nvar The number of variables per object.
     */
    inline PatchDataField(std::string name, u32 nvar)
        : field_name(std::move(name)), nvar(nvar), obj_cnt(0),
          buf(0, shamsys::instance::get_compute_scheduler_ptr()) {};

    /**
     * @brief Construct a new PatchDataField object with buffer of size obj_cnt*nvar.
     *
     * @param name The name of the field.
     * @param nvar The number of variables per object.
     * @param obj_cnt The number of object in the buffer.
     */
    inline PatchDataField(std::string name, u32 nvar, u32 obj_cnt)
        : field_name(std::move(name)), nvar(nvar), obj_cnt(obj_cnt),
          buf(obj_cnt * nvar, shamsys::instance::get_compute_scheduler_ptr()) {};

    /**
     * @brief Construct a new PatchDataField object from a moved buffer.
     *
     * @param moved_buf The buffer to move.
     * @param obj_cnt The number of object in the buffer.
     * @param name The name of the field.
     * @param nvar The number of variables per object.
     *
     * @note The buffer is moved, so the original buffer is left in an
     * uninitialized state.
     */
    inline PatchDataField(
        sham::DeviceBuffer<T> &&moved_buf, u32 obj_cnt, std::string name, u32 nvar)
        : obj_cnt(obj_cnt), field_name(name), nvar(nvar),
          buf(std::forward<sham::DeviceBuffer<T>>(moved_buf)) {}

    /**
     * @brief Construct a new PatchDataField object from a moved SYCL buffer.
     *
     * @param moved_buf The SYCL buffer to move.
     * @param obj_cnt The number of objects in the buffer.
     * @param name The name of the field.
     * @param nvar The number of variables per object.
     *
     * @note The SYCL buffer is moved, so the original buffer is left in an
     * uninitialized state.
     */
    inline PatchDataField(sycl::buffer<T> &&moved_buf, u32 obj_cnt, std::string name, u32 nvar)
        : obj_cnt(obj_cnt), field_name(name), nvar(nvar),
          buf(std::forward<sycl::buffer<T>>(moved_buf),
              obj_cnt * nvar,
              shamsys::instance::get_compute_scheduler_ptr()) {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Duplicate functions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Creates a copy of the current PatchDataField.
     *
     * @return A new PatchDataField object that is a duplicate of the current one.
     */

    inline PatchDataField duplicate() const {
        const PatchDataField &current = *this;
        return PatchDataField(current);
    }

    /**
     * @brief Creates a copy of the current PatchDataField with a new name.
     *
     * This function duplicates the current PatchDataField and assigns
     * a new name to the duplicated field.
     *
     * @param new_name The new name for the duplicated PatchDataField.
     * @return A new PatchDataField object that is a duplicate of the current
     * one but with the specified new name.
     */

    inline PatchDataField duplicate(std::string new_name) const {
        const PatchDataField &current = *this;
        PatchDataField ret            = PatchDataField(current);
        ret.field_name                = new_name;
        return ret;
    }

    /**
     * @brief Creates a copy of the current PatchDataField and returns a unique pointer to it.
     *
     * This function duplicates the current PatchDataField and returns a unique pointer
     * to the duplicated field.
     *
     * @return A unique pointer to a new PatchDataField object that is a duplicate of the current
     * one.
     */
    inline std::unique_ptr<PatchDataField> duplicate_to_ptr() const {
        const PatchDataField &current = *this;
        return std::make_unique<PatchDataField>(current);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Duplicate functions (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Data access
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Returns a reference to the internal buffer
     *
     * @return A reference to the internal buffer
     */
    inline sham::DeviceBuffer<T> &get_buf() { return buf; }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Data access (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Get the number of variables per object
    [[nodiscard]] inline const u32 &get_nvar() const { return nvar; }

    /// @brief Get the number of objects
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

    /// @brief Get the name of the field
    [[nodiscard]] inline const std::string &get_name() const { return field_name; }

    /// @brief Check if the buffer is empty
    [[nodiscard]] inline bool is_empty() const { return get_obj_cnt() == 0; }

    /// @brief Get the amount of memory used by the buffer
    [[nodiscard]] inline u64 memsize() const { return buf.get_mem_usage(); }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Getters (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Size manipulation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // every functions here should be implemented on top of resize & reserve

    /**
     * @brief Resize the buffer to accommodate a new number of objects.
     *
     * @param new_obj_cnt The new number of objects to resize the buffer to.
     */
    void resize(u32 new_obj_cnt);

    /**
     * @brief Reserve space in the buffer for a new number of objects, without changing its size.
     *
     * @param new_obj_cnt The number of objects to reserve space for.
     */
    void reserve(u32 new_obj_cnt);

    /**
     * @brief Expand the buffer by adding additional objects.
     *
     * @param obj_to_add The number of objects to add to the buffer.
     */
    void expand(u32 obj_to_add);

    /**
     * @brief Shrink the buffer by removing a number of objects.
     *
     * @param obj_to_rem The number of objects to remove from the buffer.
     */
    void shrink(u32 obj_to_rem);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Size manipulation (End)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    void insert_element(T v);

    void insert(PatchDataField<T> &f2);

    void overwrite(PatchDataField<T> &f2, u32 obj_cnt);

    void override(sycl::buffer<T> &data, u32 cnt);

    void override(std::vector<T> &data, u32 cnt);

    void override(const T val);

    inline void synchronize_buf() { buf.synchronize(); }

    void apply_offset(T off);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // get_subsets utilities
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Returns a set of all the ids of elements in the field for which the given lambda
     * evaluates to true.
     *
     * @code {.cpp}
     * std::set<u32> idx_cd = field.get_ids_set_where(
     *   [](auto access, u32 id, f64 vmin, f64 vmax) {
     *       f64 tmp = access[id];
     *       return tmp > vmin && tmp < vmax;
     *   },
     *   vmin,
     *   vmax);
     * @endcode
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
