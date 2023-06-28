// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#pragma once

#include "aliases.hpp"
#include "shamalgs/memory/serialize.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamsys/legacy/log.hpp"
#include "shambase/exception.hpp"
#include <array>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include "ResizableBuffer.hpp"


template <class T> class PatchDataField {

    ///////////////////////////////////
    // constexpr utilities (using & constexpr vals)
    ///////////////////////////////////
    static constexpr bool isprimitive = std::is_same<T, f32>::value ||
                                        std::is_same<T, f64>::value ||
                                        std::is_same<T, u32>::value || 
                                        std::is_same<T, u64>::value;

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

    using EnableIfPrimitive = enable_if_t<isprimitive>;

    using EnableIfVec = enable_if_t<is_in_type_list && (!isprimitive)>;

    constexpr static u32 min_capa  = 100;
    constexpr static f32 safe_fact = 1.25;

    ///////////////////////////////////
    // member fields
    ///////////////////////////////////

    ResizableBuffer<T> buf;

    std::string field_name;

    u32 nvar;    // number of variable per object
    u32 obj_cnt; // number of contained object



    ///////////////////////////////////
    // internal functions
    ///////////////////////////////////


    public:

    inline static sycl::buffer<T> convert_to_buf(PatchDataField<T> && pdatf){
        return ResizableBuffer<T>::convert_to_buf(std::move(pdatf.buf));
    }

    inline PatchDataField(PatchDataField &&other) noexcept
        : buf(std::move(other.buf)), field_name(std::move(other.field_name)),
          nvar(std::move(other.nvar)), obj_cnt(std::move(other.obj_cnt)) {
    } // move constructor

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
        : field_name(name), nvar(nvar), obj_cnt(0), buf(0){};

    inline PatchDataField(std::string name, u32 nvar, u32 obj_cnt)
        : field_name(name), nvar(nvar), obj_cnt(obj_cnt), buf(obj_cnt*nvar){};

    inline PatchDataField(const PatchDataField &other)
        : field_name(other.field_name), nvar(other.nvar), obj_cnt(other.obj_cnt), buf(other.buf) {
    }

    inline PatchDataField(ResizableBuffer<T> && moved_buf, u32 obj_cnt, 
    std::string name, u32 nvar) : 
        obj_cnt(obj_cnt), field_name(name),nvar(nvar), buf(std::forward<ResizableBuffer<T>>(moved_buf))
    {}

    inline PatchDataField(sycl::buffer<T> && moved_buf, u32 obj_cnt, 
    std::string name, u32 nvar) : 
        obj_cnt(obj_cnt), field_name(name),nvar(nvar), buf(std::forward<sycl::buffer<T>>(moved_buf), obj_cnt*nvar)
    {}

    static PatchDataField<T> mock_field(u64 seed, u32 obj_cnt, std::string name, u32 nvar);

    

    inline PatchDataField duplicate() const {
        const PatchDataField &current = *this;
        return PatchDataField(current);
    }

    inline PatchDataField duplicate(std::string new_name) const {
        const PatchDataField &current = *this;
        PatchDataField ret = PatchDataField(current);
        ret.field_name = new_name;
        return ret;
    }

    inline std::unique_ptr<PatchDataField> duplicate_to_ptr() const {
        const PatchDataField &current = *this;
        return std::make_unique<PatchDataField>(current);
    }

    PatchDataField &operator=(const PatchDataField &other) = delete;


    inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return buf.get_buf(); }

    inline std::unique_ptr<sycl::buffer<T>> &get_buf_priviledge() { return buf.get_buf_priviledge(); }

    //[[deprecated]]
    // inline std::unique_ptr<sycl::buffer<T>> get_sub_buf(){
    //    if(capacity > 0){
    //        return std::make_unique<sycl::buffer<T>>(*buf,0,val_cnt);
    //    }
    //    return std::unique_ptr<sycl::buffer<T>>();
    //}

    [[nodiscard]] inline const u32 &size() const { return buf.size(); }

    [[nodiscard]] inline bool is_empty() const {return size() == 0;}

    [[nodiscard]] inline u64 memsize() const { return buf.memsize(); }

    [[nodiscard]] inline const u32 &get_nvar() const { return nvar; }

    [[nodiscard]] inline const u32 &get_obj_cnt() const { return obj_cnt; }

    [[nodiscard]] inline const std::string &get_name() const { return field_name; }

    // TODO add overflow check
    void resize(u32 new_obj_cnt);

    void expand(u32 obj_to_add);

    void shrink(u32 obj_to_rem);

    void insert_element(T v);

    void apply_offset(T off);

    void insert(PatchDataField<T> &f2);

    void overwrite(PatchDataField<T> &f2, u32 obj_cnt);

    void override(sycl::buffer<T> &data, u32 cnt);

    void override(const T val);

    template <class Lambdacd>
    std::vector<u32> get_elements_with_range(Lambdacd &&cd_true, T vmin, T vmax) const;

    template <class Lambdacd>
    inline std::unique_ptr<sycl::buffer<u32>> get_elements_with_range_buf(Lambdacd &&cd_true, T vmin, T vmax) const{
        std::vector<u32> idxs = get_elements_with_range(std::forward<Lambdacd>(cd_true), vmin,vmax);
        if(idxs.empty()){
            return {};
        }else{
            return std::make_unique<sycl::buffer<u32>>(shamalgs::memory::vec_to_buf(idxs));
        }
    }

    template <class Lambdacd> void check_err_range(Lambdacd &&cd_true, T vmin, T vmax) const;

    void extract_element(u32 pidx, PatchDataField<T> &to);

    bool check_field_match(const PatchDataField<T> &f2) const;

    inline void field_raz(){
        logger::debug_ln("PatchDataField","raz : ",field_name);
        override(shambase::VectorProperties<T>::get_zero());
    }

    /**
     * @brief Copy all objects in idxs to pfield
     *
     * @param idxs
     * @param pfield
     */
    void append_subset_to(const std::vector<u32> &idxs, PatchDataField &pfield) const;
    void append_subset_to(sycl::buffer<u32> &idxs_buf, u32 sz, PatchDataField &pfield) const;

    inline PatchDataField make_new_from_subset(sycl::buffer<u32> &idxs_buf, u32 sz) const {
        PatchDataField pfield(field_name, nvar);
        append_subset_to(idxs_buf,sz,pfield);
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
     * @param len the lenght of the map (must match with the current count)
     */
    void index_remap(sycl::buffer<u32> & index_map, u32 len);

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
    void index_remap_resize(sycl::buffer<u32> & index_map, u32 len);

    /**
     * @brief minimal serialization 
     * assuming the user know the layout of the field
     * 
     * @param serializer 
     */
    void serialize_buf(shamalgs::SerializeHelper & serializer);

    /**
     * @brief deserialize a field inverse of serialize_buf
     * 
     * @param serializer 
     * @param field_name 
     * @param nvar 
     * @return PatchDataField 
     */
    static PatchDataField deserialize_buf(shamalgs::SerializeHelper & serializer, std::string field_name, u32 nvar);
    
    /**
     * @brief record the size usage of the serialization using serialize_buf
     * 
     * @return u64 
     */
    u64 serialize_buf_byte_size();

    /**
     * @brief serialize everything in the class
     * 
     * @param serializer 
     */
    void serialize_full(shamalgs::SerializeHelper & serializer);
    
    /**
     * @brief deserialize a field inverse of serialize_full
     * 
     * @param serializer 
     * @return PatchDataField 
     */
    static PatchDataField deserialize_full(shamalgs::SerializeHelper & serializer);
    
    /**
     * @brief give the size usage of serialize_full
     * 
     * @return u64 
     */
    u64 serialize_full_byte_size();

    T compute_max();
    T compute_min();
    T compute_sum();

    shambase::VecComponent<T> compute_dot_sum();

    bool has_nan();
    bool has_inf();
    bool has_nan_or_inf();
};

// TODO add overflow check
template <class T> inline void PatchDataField<T>::resize(u32 new_obj_cnt) {

    u32 new_size = new_obj_cnt * nvar;
    // field_data.resize(new_size);

    buf.resize(new_size);

    obj_cnt = new_obj_cnt;
}

template <class T> inline void PatchDataField<T>::expand(u32 obj_to_add) {
    resize(obj_cnt + obj_to_add);
}

template <class T> inline void PatchDataField<T>::shrink(u32 obj_to_rem) {

    if (obj_to_rem > obj_cnt) {
        
        throw shambase::throw_with_loc<std::invalid_argument>("impossible to remove more object than there is in the patchdata field");
    }

    resize(obj_cnt - obj_to_rem);
}

template <class T> inline void PatchDataField<T>::overwrite(PatchDataField<T> &f2, u32 obj_cnt) {
    buf.overwrite(f2.buf, obj_cnt*f2.nvar);
}

template <class T> inline void PatchDataField<T>::override(sycl::buffer<T> &data, u32 cnt) {
    buf.override(data, cnt);
}

template <class T> inline void PatchDataField<T>::override(const T val) {
    buf.override(val);
}

template <class T>
template <class Lambdacd>
inline std::vector<u32>
PatchDataField<T>::get_elements_with_range(Lambdacd &&cd_true, T vmin, T vmax) const {
    StackEntry stack_loc{};
    std::vector<u32> idxs;

    {
        sycl::host_accessor acc{*get_buf()};

        for (u32 i = 0; i < size(); i++) {
            if (cd_true(acc[i], vmin, vmax)) {
                idxs.push_back(i);
            }
        }
    }

    return idxs;
}


class PatchDataRangeCheckError : public std::exception {
  public:
    explicit PatchDataRangeCheckError(const char *message) : msg_(message) {}

    explicit PatchDataRangeCheckError(const std::string &message) : msg_(message) {}

     ~PatchDataRangeCheckError() noexcept override = default;

    [[nodiscard]] 
     const char *what() const noexcept override { return msg_.c_str(); }

  protected:
    std::string msg_;
};

template <class T>
template <class Lambdacd>
inline void PatchDataField<T>::check_err_range(Lambdacd &&cd_true, T vmin, T vmax) const {

    if(is_empty()){return;}

    bool error = false;
    {
        sycl::host_accessor acc{*get_buf()};
        u32 err_cnt = 0;

        for (u32 i = 0; i < size(); i++) {
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
                    "]"
                );
                error = true;
                err_cnt ++;
                if(err_cnt > 50){
                    logger::err_ln(
                        "PatchDataField",
                        "..."
                    );
                    break;
                }
            }
        }
    }

    if(error){
        throw shambase::throw_with_loc<PatchDataRangeCheckError>("obj not in range");
    }

}