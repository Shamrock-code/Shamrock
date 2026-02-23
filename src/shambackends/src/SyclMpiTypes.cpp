// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SyclMpiTypes.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"

bool __mpi_sycl_type_active = false;
bool is_mpi_sycl_interop_active() { return __mpi_sycl_type_active; }

/*
const int __len_vec2  [] = {1,1};
const int __len_vec3  [] = {1,1,1};
const int __len_vec4  [] = {1,1,1,1};
const int __len_vec8  [] = {1,1,1,1,1,1,1,1};
const int __len_vec16 [] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
*/
const int __len_vec2  = 2;
const int __len_vec3  = 3;
const int __len_vec4  = 4;
const int __len_vec8  = 8;
const int __len_vec16 = 16;

inline MPI_Datatype __tmp_mpi_type_i64_3;
inline MPI_Datatype __tmp_mpi_type_i32_3;
inline MPI_Datatype __tmp_mpi_type_i16_3;
inline MPI_Datatype __tmp_mpi_type_i8_3;
inline MPI_Datatype __tmp_mpi_type_u64_3;
inline MPI_Datatype __tmp_mpi_type_u32_3;
inline MPI_Datatype __tmp_mpi_type_u16_3;
inline MPI_Datatype __tmp_mpi_type_u8_3;
inline MPI_Datatype __tmp_mpi_type_f16_3;
inline MPI_Datatype __tmp_mpi_type_f32_3;
inline MPI_Datatype __tmp_mpi_type_f64_3;

#define __SYCL_TYPE_COMMIT_len2(base_name, src_type)                                               \
    {                                                                                              \
        check_offset_validity<base_name>();                                                        \
        MPICHECK(MPI_Type_contiguous(__len_vec2, mpi_type_##src_type, &mpi_type_##base_name));     \
        MPICHECK(MPI_Type_commit(&mpi_type_##base_name));                                          \
        shamlog_debug_mpi_ln("SyclMpiTypes", "init mpi type for : " #base_name);                   \
    }

#define __SYCL_TYPE_COMMIT_len3(base_name, src_type)                                               \
    {                                                                                              \
        check_offset_validity<base_name>();                                                        \
        MPICHECK(                                                                                  \
            MPI_Type_contiguous(__len_vec3, mpi_type_##src_type, &__tmp_mpi_type_##base_name));    \
        MPICHECK(MPI_Type_create_resized(                                                          \
            __tmp_mpi_type_##base_name, 0, sizeof(base_name), &mpi_type_##base_name));             \
        MPICHECK(MPI_Type_commit(&mpi_type_##base_name));                                          \
        shamlog_debug_mpi_ln("SyclMpiTypes", "init mpi type for : " #base_name);                   \
    }

#define __SYCL_TYPE_COMMIT_len4(base_name, src_type)                                               \
    {                                                                                              \
        check_offset_validity<base_name>();                                                        \
        MPICHECK(MPI_Type_contiguous(__len_vec4, mpi_type_##src_type, &mpi_type_##base_name));     \
        MPICHECK(MPI_Type_commit(&mpi_type_##base_name));                                          \
        shamlog_debug_mpi_ln("SyclMpiTypes", "init mpi type for : " #base_name);                   \
    }

#define __SYCL_TYPE_COMMIT_len8(base_name, src_type)                                               \
    {                                                                                              \
        check_offset_validity<base_name>();                                                        \
        MPICHECK(MPI_Type_contiguous(__len_vec8, mpi_type_##src_type, &mpi_type_##base_name));     \
        MPICHECK(MPI_Type_commit(&mpi_type_##base_name));                                          \
        shamlog_debug_mpi_ln("SyclMpiTypes", "init mpi type for : " #base_name);                   \
    }

#define __SYCL_TYPE_COMMIT_len16(base_name, src_type)                                              \
    {                                                                                              \
        check_offset_validity<base_name>();                                                        \
        MPICHECK(MPI_Type_contiguous(__len_vec16, mpi_type_##src_type, &mpi_type_##base_name));    \
        MPICHECK(MPI_Type_commit(&mpi_type_##base_name));                                          \
        shamlog_debug_mpi_ln("SyclMpiTypes", "init mpi type for : " #base_name);                   \
    }

template<class T>
void check_offset_validity() {
    T a{};

    std::ptrdiff_t base = reinterpret_cast<std::ptrdiff_t>(&a);
    std::ptrdiff_t s0   = reinterpret_cast<std::ptrdiff_t>(&a.s0());

    if (s0 - base != 0) {
        throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
            "Offset is not valid for type {}, base = {}, s0 = {}", typeid(T).name(), base, s0));
    }
}

void create_sycl_mpi_types() {

    __SYCL_TYPE_COMMIT_len2(i64_2, i64);
    __SYCL_TYPE_COMMIT_len2(i32_2, i32);
    __SYCL_TYPE_COMMIT_len2(i16_2, i16);
    __SYCL_TYPE_COMMIT_len2(i8_2, i8);
    __SYCL_TYPE_COMMIT_len2(u64_2, u64);
    __SYCL_TYPE_COMMIT_len2(u32_2, u32);
    __SYCL_TYPE_COMMIT_len2(u16_2, u16);
    __SYCL_TYPE_COMMIT_len2(u8_2, u8);
    __SYCL_TYPE_COMMIT_len2(f16_2, f16);
    __SYCL_TYPE_COMMIT_len2(f32_2, f32);
    __SYCL_TYPE_COMMIT_len2(f64_2, f64);

    __SYCL_TYPE_COMMIT_len3(i64_3, i64);
    __SYCL_TYPE_COMMIT_len3(i32_3, i32);
    __SYCL_TYPE_COMMIT_len3(i16_3, i16);

    // {
    //     i16_3 a;

    //     MPI_Datatype types_list[3] = {mpi_type_i16,mpi_type_i16,mpi_type_i16};
    //     int          block_lens[3] = {1,1,1};
    //     MPI_Aint     MPI_offset[3];
    //        MPI_offset[0] = ((size_t) ( (char *)&(a.x()) - (char *)&(a) ));
    //        MPI_offset[1] = ((size_t) ( (char *)&(a.y()) - (char *)&(a) ));
    //        MPI_offset[2] = ((size_t) ( (char *)&(a.z()) - (char *)&(a) ));

    //     mpi::type_create_struct( 3,  block_lens, MPI_offset, types_list, &mpi_type_i16_3 );
    //     /*mpi::type_create_resized(__tmp_mpi_type_i16_3, 0, sizeof(base_name),
    //     &mpi_type_i16_3);*/\ mpi::type_commit( &mpi_type_i16_3 );
    // }

    __SYCL_TYPE_COMMIT_len3(i8_3, i8);
    __SYCL_TYPE_COMMIT_len3(u64_3, u64);
    __SYCL_TYPE_COMMIT_len3(u32_3, u32);
    __SYCL_TYPE_COMMIT_len3(u16_3, u16);
    __SYCL_TYPE_COMMIT_len3(u8_3, u8);
    __SYCL_TYPE_COMMIT_len3(f16_3, f16);
    __SYCL_TYPE_COMMIT_len3(f32_3, f32);
    __SYCL_TYPE_COMMIT_len3(f64_3, f64);

    __SYCL_TYPE_COMMIT_len4(i64_4, i64);
    __SYCL_TYPE_COMMIT_len4(i32_4, i32);
    __SYCL_TYPE_COMMIT_len4(i16_4, i16);
    __SYCL_TYPE_COMMIT_len4(i8_4, i8);
    __SYCL_TYPE_COMMIT_len4(u64_4, u64);
    __SYCL_TYPE_COMMIT_len4(u32_4, u32);
    __SYCL_TYPE_COMMIT_len4(u16_4, u16);
    __SYCL_TYPE_COMMIT_len4(u8_4, u8);
    __SYCL_TYPE_COMMIT_len4(f16_4, f16);
    __SYCL_TYPE_COMMIT_len4(f32_4, f32);
    __SYCL_TYPE_COMMIT_len4(f64_4, f64);

    __SYCL_TYPE_COMMIT_len8(i64_8, i64);
    __SYCL_TYPE_COMMIT_len8(i32_8, i32);
    __SYCL_TYPE_COMMIT_len8(i16_8, i16);
    __SYCL_TYPE_COMMIT_len8(i8_8, i8);
    __SYCL_TYPE_COMMIT_len8(u64_8, u64);
    __SYCL_TYPE_COMMIT_len8(u32_8, u32);
    __SYCL_TYPE_COMMIT_len8(u16_8, u16);
    __SYCL_TYPE_COMMIT_len8(u8_8, u8);
    __SYCL_TYPE_COMMIT_len8(f16_8, f16);
    __SYCL_TYPE_COMMIT_len8(f32_8, f32);
    __SYCL_TYPE_COMMIT_len8(f64_8, f64);

    __SYCL_TYPE_COMMIT_len16(i64_16, i64);
    __SYCL_TYPE_COMMIT_len16(i32_16, i32);
    __SYCL_TYPE_COMMIT_len16(i16_16, i16);
    __SYCL_TYPE_COMMIT_len16(i8_16, i8);
    __SYCL_TYPE_COMMIT_len16(u64_16, u64);
    __SYCL_TYPE_COMMIT_len16(u32_16, u32);
    __SYCL_TYPE_COMMIT_len16(u16_16, u16);
    __SYCL_TYPE_COMMIT_len16(u8_16, u8);
    __SYCL_TYPE_COMMIT_len16(f16_16, f16);
    __SYCL_TYPE_COMMIT_len16(f32_16, f32);
    __SYCL_TYPE_COMMIT_len16(f64_16, f64);

    __mpi_sycl_type_active = true;
}

void free_sycl_mpi_types() {

    MPICHECK(MPI_Type_free(&mpi_type_i64_2));
    MPICHECK(MPI_Type_free(&mpi_type_i32_2));
    MPICHECK(MPI_Type_free(&mpi_type_i16_2));
    MPICHECK(MPI_Type_free(&mpi_type_i8_2));
    MPICHECK(MPI_Type_free(&mpi_type_u64_2));
    MPICHECK(MPI_Type_free(&mpi_type_u32_2));
    MPICHECK(MPI_Type_free(&mpi_type_u16_2));
    MPICHECK(MPI_Type_free(&mpi_type_u8_2));
    MPICHECK(MPI_Type_free(&mpi_type_f16_2));
    MPICHECK(MPI_Type_free(&mpi_type_f32_2));
    MPICHECK(MPI_Type_free(&mpi_type_f64_2));

    MPICHECK(MPI_Type_free(&mpi_type_i64_3));
    MPICHECK(MPI_Type_free(&mpi_type_i32_3));
    MPICHECK(MPI_Type_free(&mpi_type_i16_3));
    MPICHECK(MPI_Type_free(&mpi_type_i8_3));
    MPICHECK(MPI_Type_free(&mpi_type_u64_3));
    MPICHECK(MPI_Type_free(&mpi_type_u32_3));
    MPICHECK(MPI_Type_free(&mpi_type_u16_3));
    MPICHECK(MPI_Type_free(&mpi_type_u8_3));
    MPICHECK(MPI_Type_free(&mpi_type_f16_3));
    MPICHECK(MPI_Type_free(&mpi_type_f32_3));
    MPICHECK(MPI_Type_free(&mpi_type_f64_3));

    MPICHECK(MPI_Type_free(&mpi_type_i64_4));
    MPICHECK(MPI_Type_free(&mpi_type_i32_4));
    MPICHECK(MPI_Type_free(&mpi_type_i16_4));
    MPICHECK(MPI_Type_free(&mpi_type_i8_4));
    MPICHECK(MPI_Type_free(&mpi_type_u64_4));
    MPICHECK(MPI_Type_free(&mpi_type_u32_4));
    MPICHECK(MPI_Type_free(&mpi_type_u16_4));
    MPICHECK(MPI_Type_free(&mpi_type_u8_4));
    MPICHECK(MPI_Type_free(&mpi_type_f16_4));
    MPICHECK(MPI_Type_free(&mpi_type_f32_4));
    MPICHECK(MPI_Type_free(&mpi_type_f64_4));

    MPICHECK(MPI_Type_free(&mpi_type_i64_8));
    MPICHECK(MPI_Type_free(&mpi_type_i32_8));
    MPICHECK(MPI_Type_free(&mpi_type_i16_8));
    MPICHECK(MPI_Type_free(&mpi_type_i8_8));
    MPICHECK(MPI_Type_free(&mpi_type_u64_8));
    MPICHECK(MPI_Type_free(&mpi_type_u32_8));
    MPICHECK(MPI_Type_free(&mpi_type_u16_8));
    MPICHECK(MPI_Type_free(&mpi_type_u8_8));
    MPICHECK(MPI_Type_free(&mpi_type_f16_8));
    MPICHECK(MPI_Type_free(&mpi_type_f32_8));
    MPICHECK(MPI_Type_free(&mpi_type_f64_8));

    MPICHECK(MPI_Type_free(&mpi_type_i64_16));
    MPICHECK(MPI_Type_free(&mpi_type_i32_16));
    MPICHECK(MPI_Type_free(&mpi_type_i16_16));
    MPICHECK(MPI_Type_free(&mpi_type_i8_16));
    MPICHECK(MPI_Type_free(&mpi_type_u64_16));
    MPICHECK(MPI_Type_free(&mpi_type_u32_16));
    MPICHECK(MPI_Type_free(&mpi_type_u16_16));
    MPICHECK(MPI_Type_free(&mpi_type_u8_16));
    MPICHECK(MPI_Type_free(&mpi_type_f16_16));
    MPICHECK(MPI_Type_free(&mpi_type_f32_16));
    MPICHECK(MPI_Type_free(&mpi_type_f64_16));

    __mpi_sycl_type_active = false;
}
