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
#include "shamalgs/algorithm.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/serialize.hpp"
#include "shambase/exception.hpp"

template<class T>
class ResizableBuffer {

    std::unique_ptr<sycl::buffer<T>> buf;

    u32 capacity;
    u32 val_cnt; // nvar*obj_cnt

    constexpr static u32 min_capa  = 100;
    constexpr static f32 safe_fact = 1.25;

    void alloc();
    void free();
    void change_capacity(u32 new_capa);

    public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // memory manipulation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Get the reference to the internal buffer
     *
     * @return const std::unique_ptr<sycl::buffer<T>>&
     */
    inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return buf; }

    /**
     * @brief Get a editable reference to the internal buffer
     *
     * @return const std::unique_ptr<sycl::buffer<T>>&
     */
    inline std::unique_ptr<sycl::buffer<T>> &get_buf_priviledge() { return buf; }

    /**
     * @brief get the number of held values in the buffer
     *
     * @return const u32&
     */
    [[nodiscard]] inline const u32 &size() const { return val_cnt; }

    /**
     * @brief get the memsize of the buffer
     *
     * @return u64
     */
    [[nodiscard]] inline u64 memsize() const { return val_cnt * sizeof(T); }

    /**
     * @brief resize the buffer size
     *
     * @param new_size
     */
    void resize(u32 new_size);

    /**
     * @brief reserve slots for the buffer the buffer size
     *
     * @param add_size
     */
    void reserve(u32 add_size);


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // value manipulation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    void overwrite(ResizableBuffer<T> &f2, u32 cnt);

    void override(sycl::buffer<T> &data, u32 cnt);

    void override(const T val);

    void index_remap_resize(sycl::buffer<u32> &index_map, u32 len, u32 nvar = 1);

    [[nodiscard]] bool check_buf_match(const ResizableBuffer<T> &f2) const;


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // constructors & destructors
    ////////////////////////////////////////////////////////////////////////////////////////////////

    explicit ResizableBuffer(u32 cnt = 0) : val_cnt(cnt), capacity(cnt) {
        if (capacity != 0) {
            alloc();
        }
    }

    inline ResizableBuffer(sycl::buffer<T> &&moved_buf, u32 val_cnt)
        : buf(std::make_unique<sycl::buffer<T>>(std::move(moved_buf))), val_cnt(val_cnt),
          capacity(moved_buf.size()) {}

    ResizableBuffer(const ResizableBuffer &other)
        : val_cnt(other.val_cnt), capacity(other.capacity) {
        if (capacity != 0) {
            alloc();
            // copydata(other._data,_data, capacity);
            shamalgs::memory::copybuf_discard(*other.buf, *buf, capacity);
        }
    }

    ResizableBuffer(ResizableBuffer &&other) noexcept
        : buf(std::move(other.buf)), val_cnt(std::move(other.val_cnt)),
          capacity(std::move(other.capacity)) {} // move constructor

    ResizableBuffer &operator=(ResizableBuffer &&other) noexcept {
        buf      = std::move(other.buf);
        val_cnt  = std::move(other.val_cnt);
        capacity = std::move(other.capacity);

        return *this;
    } // move assignment

    /**
     * @brief serialize the content of the buffer
     *  Note : no size information will be written
     * @param serializer 
     */
    void serialize_buf(shamalgs::SerializeHelper & serializer);

    /**
     * @brief inverse operation of @serialize_buf
     * 
     * @param serializer 
     * @param val_cnt 
     * @return ResizableBuffer 
     */
    static ResizableBuffer deserialize_buf (shamalgs::SerializeHelper & serializer, u32 val_cnt);

    u64 serialize_buf_byte_size();

    static ResizableBuffer mock_buffer(u64 seed, u32 val_cnt, T min_bound, T max_bound);

    ResizableBuffer &operator=(const ResizableBuffer &other) // copy assignment
        = delete;

    inline ~ResizableBuffer() { free(); }
};
