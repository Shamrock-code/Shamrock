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
 * @file bytestream.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace shambase {

    template<class T>
    inline void stream_write(std::basic_stringstream<byte> &stream, T &obj) {
        stream.write(reinterpret_cast<byte const *>(&obj), sizeof(obj));
    }

    template<class T>
    inline void stream_read(std::basic_stringstream<byte> &stream, T &obj) {
        stream.read(reinterpret_cast<byte *>(&obj), sizeof(obj));
    }

    inline void stream_write_string(std::basic_stringstream<byte> &stream, std::string &s) {
        u64 strlen = s.size();
        stream_write(stream, strlen);
        stream.write(reinterpret_cast<byte const *>(s.data()), strlen * sizeof(char));
    }

    inline void stream_read_string(std::basic_stringstream<byte> &stream, std::string &s) {
        u64 strlen;
        stream_read<u64>(stream, strlen);
        s.resize(strlen);
        stream.read(reinterpret_cast<byte *>(s.data()), strlen * sizeof(char));
    }

    namespace details {
        template<typename T>
        using serialize_t = decltype(std::declval<T>().serialize(
            std::declval<std::basic_stringstream<byte> &>()));

        template<typename Container, typename = std::void_t<>>
        struct has_serialize : std::false_type {};

        template<typename Container>
        struct has_serialize<Container, std::void_t<serialize_t<Container>>> : std::true_type {};

        template<typename Container>
        inline constexpr bool has_serialize_v = has_serialize<Container>::value;

        template<typename T>
        using deserialize_t
            = decltype(T::deserialize(std::declval<std::basic_stringstream<byte> &>()));

        template<typename Container, typename = std::void_t<>>
        struct has_deserialize : std::false_type {};

        template<typename Container>
        struct has_deserialize<Container, std::void_t<deserialize_t<Container>>> : std::true_type {
        };

        template<typename Container>
        inline constexpr bool has_deserialize_v = has_deserialize<Container>::value;

        enum VALIDATION_FLAGS {
            VECTOR = 0x124,
        };

        class bytestreamException : public std::exception {
            public:
            explicit bytestreamException(const char *message) : msg_(message) {}

            explicit bytestreamException(const std::string &message) : msg_(message) {}

            virtual ~bytestreamException() noexcept {}

            virtual const char *what() const noexcept { return msg_.c_str(); }

            protected:
            std::string msg_;
        };
    } // namespace details

    /**
     * @brief write the vector into the bytestream
     *
     * @tparam T
     * @param stream
     * @param vec
     */
    template<class T>
    inline void stream_write_vector(std::basic_stringstream<byte> &stream, std::vector<T> &vec) {

        static_assert(
            details::has_serialize_v<T>, "the template class must have serialize implemented");

        u32 flag = details::VALIDATION_FLAGS::VECTOR;
        stream_write(stream, flag);

        u64 len = vec.size();
        stream_write(stream, len);
        for (T &v : vec) {
            v.serialize(stream);
        }
    }

    /**
     * @brief read a vector from the bytestream
     * Note : this appends read objects to the vector without resetting it
     *
     * @tparam T
     * @param stream
     * @param vec
     */
    template<class T>
    inline void stream_read_vector(std::basic_stringstream<byte> &stream, std::vector<T> &vec) {
        static_assert(
            details::has_serialize_v<T>, "the template class must have deserialize implemented");

        u32 flag;
        stream_read(stream, flag);
        if (flag != details::VALIDATION_FLAGS::VECTOR) {
            throw details::bytestreamException("the validation flags don't match");
        }

        u64 len;
        stream_read(stream, len);
        for (u64 i = 0; i < len; i++) {
            vec.push_back(T::deserialize(stream));
        }
    }

    template<class T>
    inline void stream_write_vector_trivial(
        std::basic_stringstream<byte> &stream, std::vector<T> &vec) {

        u32 flag = details::VALIDATION_FLAGS::VECTOR;
        stream_write(stream, flag);

        u64 len = vec.size();
        stream_write(stream, len);

        stream.write(reinterpret_cast<byte const *>(vec.data()), len * sizeof(T));
    }

    /**
     * @brief read a vector from the bytestream
     * Note : this appends read objects to the vector without resetting it
     *
     * @tparam T
     * @param stream
     * @param vec
     */
    template<class T>
    inline void stream_read_vector_trivial(
        std::basic_stringstream<byte> &stream, std::vector<T> &vec) {

        u32 flag;
        stream_read(stream, flag);
        if (flag != details::VALIDATION_FLAGS::VECTOR) {
            throw details::bytestreamException("the validation flags don't match");
        }

        u64 len;
        stream_read(stream, len);
        vec.resize(len);

        stream.read(reinterpret_cast<byte *>(vec.data()), len * sizeof(T));
    }

} // namespace shambase
