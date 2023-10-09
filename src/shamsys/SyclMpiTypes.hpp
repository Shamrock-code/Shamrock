// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SyclMpiTypes.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#include "shamsys/MpiWrapper.hpp"
#include "shamsys/legacy/log.hpp"

#include "shambase/sycl_vec_aliases.hpp"





inline MPI_Datatype mpi_type_i64 = MPI_INT64_T;
inline MPI_Datatype mpi_type_i32 = MPI_INT32_T;
inline MPI_Datatype mpi_type_i16 = MPI_INT16_T;
inline MPI_Datatype mpi_type_i8  = MPI_INT8_T ;
inline MPI_Datatype mpi_type_u64 = MPI_UINT64_T;
inline MPI_Datatype mpi_type_u32 = MPI_UINT32_T;
inline MPI_Datatype mpi_type_u16 = MPI_UINT16_T;
inline MPI_Datatype mpi_type_u8  = MPI_UINT8_T ;
inline MPI_Datatype mpi_type_f16 = MPI_SHORT; // no f16 in mpi std
inline MPI_Datatype mpi_type_f32 = MPI_FLOAT;
inline MPI_Datatype mpi_type_f64 = MPI_DOUBLE;


inline MPI_Datatype mpi_type_i64_2;
inline MPI_Datatype mpi_type_i32_2;
inline MPI_Datatype mpi_type_i16_2;
inline MPI_Datatype mpi_type_i8_2 ;
inline MPI_Datatype mpi_type_u64_2;
inline MPI_Datatype mpi_type_u32_2;
inline MPI_Datatype mpi_type_u16_2;
inline MPI_Datatype mpi_type_u8_2 ;
inline MPI_Datatype mpi_type_f16_2;
inline MPI_Datatype mpi_type_f32_2;
inline MPI_Datatype mpi_type_f64_2;

inline MPI_Datatype mpi_type_i64_3;
inline MPI_Datatype mpi_type_i32_3;
inline MPI_Datatype mpi_type_i16_3;
inline MPI_Datatype mpi_type_i8_3 ; 
inline MPI_Datatype mpi_type_u64_3;
inline MPI_Datatype mpi_type_u32_3;
inline MPI_Datatype mpi_type_u16_3;
inline MPI_Datatype mpi_type_u8_3 ; 
inline MPI_Datatype mpi_type_f16_3;
inline MPI_Datatype mpi_type_f32_3;
inline MPI_Datatype mpi_type_f64_3;

inline MPI_Datatype mpi_type_i64_4;
inline MPI_Datatype mpi_type_i32_4;
inline MPI_Datatype mpi_type_i16_4;
inline MPI_Datatype mpi_type_i8_4 ; 
inline MPI_Datatype mpi_type_u64_4;
inline MPI_Datatype mpi_type_u32_4;
inline MPI_Datatype mpi_type_u16_4;
inline MPI_Datatype mpi_type_u8_4 ; 
inline MPI_Datatype mpi_type_f16_4;
inline MPI_Datatype mpi_type_f32_4;
inline MPI_Datatype mpi_type_f64_4;

inline MPI_Datatype mpi_type_i64_8;
inline MPI_Datatype mpi_type_i32_8;
inline MPI_Datatype mpi_type_i16_8;
inline MPI_Datatype mpi_type_i8_8 ;
inline MPI_Datatype mpi_type_u64_8;
inline MPI_Datatype mpi_type_u32_8;
inline MPI_Datatype mpi_type_u16_8;
inline MPI_Datatype mpi_type_u8_8 ;
inline MPI_Datatype mpi_type_f16_8;
inline MPI_Datatype mpi_type_f32_8;
inline MPI_Datatype mpi_type_f64_8;

inline MPI_Datatype mpi_type_i64_16;
inline MPI_Datatype mpi_type_i32_16;
inline MPI_Datatype mpi_type_i16_16;
inline MPI_Datatype mpi_type_i8_16 ;
inline MPI_Datatype mpi_type_u64_16;
inline MPI_Datatype mpi_type_u32_16;
inline MPI_Datatype mpi_type_u16_16;
inline MPI_Datatype mpi_type_u8_16 ;
inline MPI_Datatype mpi_type_f16_16;
inline MPI_Datatype mpi_type_f32_16;
inline MPI_Datatype mpi_type_f64_16;


template<class type>
MPI_Datatype & get_mpi_type(); 

//coments due to weird implementation of half prec in hipsycl

template<> inline MPI_Datatype & get_mpi_type<i64      >(){return mpi_type_i64    ;}
template<> inline MPI_Datatype & get_mpi_type<i32      >(){return mpi_type_i32    ;}
template<> inline MPI_Datatype & get_mpi_type<i16      >(){return mpi_type_i16    ;}
template<> inline MPI_Datatype & get_mpi_type<i8       >(){return mpi_type_i8     ;}
template<> inline MPI_Datatype & get_mpi_type<u64      >(){return mpi_type_u64    ;}
template<> inline MPI_Datatype & get_mpi_type<u32      >(){return mpi_type_u32    ;}
template<> inline MPI_Datatype & get_mpi_type<u16      >(){return mpi_type_u16    ;}
template<> inline MPI_Datatype & get_mpi_type<u8       >(){return mpi_type_u8     ;}
//template<> inline MPI_Datatype & get_mpi_type<f16      >(){return mpi_type_f16    ;}
template<> inline MPI_Datatype & get_mpi_type<f32      >(){return mpi_type_f32    ;}
template<> inline MPI_Datatype & get_mpi_type<f64      >(){return mpi_type_f64    ;}

template<> inline MPI_Datatype & get_mpi_type<i64_2    >(){return mpi_type_i64_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_2    >(){return mpi_type_i32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_2    >(){return mpi_type_i16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_2     >(){return mpi_type_i8_2   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_2    >(){return mpi_type_u64_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_2    >(){return mpi_type_u32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_2    >(){return mpi_type_u16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_2     >(){return mpi_type_u8_2   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_2    >(){return mpi_type_f16_2  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_2    >(){return mpi_type_f32_2  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_2    >(){return mpi_type_f64_2  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_3    >(){return mpi_type_i64_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_3    >(){return mpi_type_i32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_3    >(){return mpi_type_i16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_3     >(){return mpi_type_i8_3   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_3    >(){return mpi_type_u64_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_3    >(){return mpi_type_u32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_3    >(){return mpi_type_u16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_3     >(){return mpi_type_u8_3   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_3    >(){return mpi_type_f16_3  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_3    >(){return mpi_type_f32_3  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_3    >(){return mpi_type_f64_3  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_4    >(){return mpi_type_i64_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_4    >(){return mpi_type_i32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_4    >(){return mpi_type_i16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_4     >(){return mpi_type_i8_4   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_4    >(){return mpi_type_u64_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_4    >(){return mpi_type_u32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_4    >(){return mpi_type_u16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_4     >(){return mpi_type_u8_4   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_4    >(){return mpi_type_f16_4  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_4    >(){return mpi_type_f32_4  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_4    >(){return mpi_type_f64_4  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_8    >(){return mpi_type_i64_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i32_8    >(){return mpi_type_i32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i16_8    >(){return mpi_type_i16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<i8_8     >(){return mpi_type_i8_8   ;}
template<> inline MPI_Datatype & get_mpi_type<u64_8    >(){return mpi_type_u64_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u32_8    >(){return mpi_type_u32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u16_8    >(){return mpi_type_u16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<u8_8     >(){return mpi_type_u8_8   ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_8    >(){return mpi_type_f16_8  ;}
template<> inline MPI_Datatype & get_mpi_type<f32_8    >(){return mpi_type_f32_8  ;}
template<> inline MPI_Datatype & get_mpi_type<f64_8    >(){return mpi_type_f64_8  ;}

template<> inline MPI_Datatype & get_mpi_type<i64_16   >(){return mpi_type_i64_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i32_16   >(){return mpi_type_i32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i16_16   >(){return mpi_type_i16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<i8_16    >(){return mpi_type_i8_16  ;}
template<> inline MPI_Datatype & get_mpi_type<u64_16   >(){return mpi_type_u64_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u32_16   >(){return mpi_type_u32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u16_16   >(){return mpi_type_u16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<u8_16    >(){return mpi_type_u8_16  ;}
//template<> inline MPI_Datatype & get_mpi_type<f16_16   >(){return mpi_type_f16_16 ;}
template<> inline MPI_Datatype & get_mpi_type<f32_16   >(){return mpi_type_f32_16 ;}
template<> inline MPI_Datatype & get_mpi_type<f64_16   >(){return mpi_type_f64_16 ;}


void create_sycl_mpi_types();
void free_sycl_mpi_types();
bool is_mpi_sycl_interop_active();