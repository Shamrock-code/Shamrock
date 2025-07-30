// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/mpiErrorCheck.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

int get_mpi_size(MPI_Datatype md) {
    MPI_Aint lbs;
    MPI_Aint exts;

    MPICHECK(MPI_Type_get_extent(md, &lbs, &exts));
    return exts;
}

TestStart(Unittest, "syclhandler/test_sycl_mpi_types_sizes", test_sycl_mpi_types_sizes, -1) {

    REQUIRE_EQUAL_NAMED("mpi type i64    correct", get_mpi_size(mpi_type_i64), sizeof(i64));
    REQUIRE_EQUAL_NAMED("mpi type i32    correct", get_mpi_size(mpi_type_i32), sizeof(i32));
    REQUIRE_EQUAL_NAMED("mpi type i16    correct", get_mpi_size(mpi_type_i16), sizeof(i16));
    REQUIRE_EQUAL_NAMED("mpi type i8     correct", get_mpi_size(mpi_type_i8), sizeof(i8));
    REQUIRE_EQUAL_NAMED("mpi type u64    correct", get_mpi_size(mpi_type_u64), sizeof(u64));
    REQUIRE_EQUAL_NAMED("mpi type u32    correct", get_mpi_size(mpi_type_u32), sizeof(u32));
    REQUIRE_EQUAL_NAMED("mpi type u16    correct", get_mpi_size(mpi_type_u16), sizeof(u16));
    REQUIRE_EQUAL_NAMED("mpi type u8     correct", get_mpi_size(mpi_type_u8), sizeof(u8));
    REQUIRE_EQUAL_NAMED("mpi type f16    correct", get_mpi_size(mpi_type_f16), sizeof(f16));
    REQUIRE_EQUAL_NAMED("mpi type f32    correct", get_mpi_size(mpi_type_f32), sizeof(f32));
    REQUIRE_EQUAL_NAMED("mpi type f64    correct", get_mpi_size(mpi_type_f64), sizeof(f64));
    REQUIRE_EQUAL_NAMED("mpi type i64_2  correct", get_mpi_size(mpi_type_i64_2), sizeof(i64_2));
    REQUIRE_EQUAL_NAMED("mpi type i32_2  correct", get_mpi_size(mpi_type_i32_2), sizeof(i32_2));
    REQUIRE_EQUAL_NAMED("mpi type i16_2  correct", get_mpi_size(mpi_type_i16_2), sizeof(i16_2));
    REQUIRE_EQUAL_NAMED("mpi type i8_2   correct", get_mpi_size(mpi_type_i8_2), sizeof(i8_2));
    REQUIRE_EQUAL_NAMED("mpi type u64_2  correct", get_mpi_size(mpi_type_u64_2), sizeof(u64_2));
    REQUIRE_EQUAL_NAMED("mpi type u32_2  correct", get_mpi_size(mpi_type_u32_2), sizeof(u32_2));
    REQUIRE_EQUAL_NAMED("mpi type u16_2  correct", get_mpi_size(mpi_type_u16_2), sizeof(u16_2));
    REQUIRE_EQUAL_NAMED("mpi type u8_2   correct", get_mpi_size(mpi_type_u8_2), sizeof(u8_2));
    REQUIRE_EQUAL_NAMED("mpi type f16_2  correct", get_mpi_size(mpi_type_f16_2), sizeof(f16_2));
    REQUIRE_EQUAL_NAMED("mpi type f32_2  correct", get_mpi_size(mpi_type_f32_2), sizeof(f32_2));
    REQUIRE_EQUAL_NAMED("mpi type f64_2  correct", get_mpi_size(mpi_type_f64_2), sizeof(f64_2));
    REQUIRE_EQUAL_NAMED("mpi type i64_3  correct", get_mpi_size(mpi_type_i64_3), sizeof(i64_3));
    REQUIRE_EQUAL_NAMED("mpi type i32_3  correct", get_mpi_size(mpi_type_i32_3), sizeof(i32_3));
    REQUIRE_EQUAL_NAMED("mpi type i16_3  correct", get_mpi_size(mpi_type_i16_3), sizeof(i16_3));
    REQUIRE_EQUAL_NAMED("mpi type i8_3   correct", get_mpi_size(mpi_type_i8_3), sizeof(i8_3));
    REQUIRE_EQUAL_NAMED("mpi type u64_3  correct", get_mpi_size(mpi_type_u64_3), sizeof(u64_3));
    REQUIRE_EQUAL_NAMED("mpi type u32_3  correct", get_mpi_size(mpi_type_u32_3), sizeof(u32_3));
    REQUIRE_EQUAL_NAMED("mpi type u16_3  correct", get_mpi_size(mpi_type_u16_3), sizeof(u16_3));
    REQUIRE_EQUAL_NAMED("mpi type u8_3   correct", get_mpi_size(mpi_type_u8_3), sizeof(u8_3));
    REQUIRE_EQUAL_NAMED("mpi type f16_3  correct", get_mpi_size(mpi_type_f16_3), sizeof(f16_3));
    REQUIRE_EQUAL_NAMED("mpi type f32_3  correct", get_mpi_size(mpi_type_f32_3), sizeof(f32_3));
    REQUIRE_EQUAL_NAMED("mpi type f64_3  correct", get_mpi_size(mpi_type_f64_3), sizeof(f64_3));
    REQUIRE_EQUAL_NAMED("mpi type i64_4  correct", get_mpi_size(mpi_type_i64_4), sizeof(i64_4));
    REQUIRE_EQUAL_NAMED("mpi type i32_4  correct", get_mpi_size(mpi_type_i32_4), sizeof(i32_4));
    REQUIRE_EQUAL_NAMED("mpi type i16_4  correct", get_mpi_size(mpi_type_i16_4), sizeof(i16_4));
    REQUIRE_EQUAL_NAMED("mpi type i8_4   correct", get_mpi_size(mpi_type_i8_4), sizeof(i8_4));
    REQUIRE_EQUAL_NAMED("mpi type u64_4  correct", get_mpi_size(mpi_type_u64_4), sizeof(u64_4));
    REQUIRE_EQUAL_NAMED("mpi type u32_4  correct", get_mpi_size(mpi_type_u32_4), sizeof(u32_4));
    REQUIRE_EQUAL_NAMED("mpi type u16_4  correct", get_mpi_size(mpi_type_u16_4), sizeof(u16_4));
    REQUIRE_EQUAL_NAMED("mpi type u8_4   correct", get_mpi_size(mpi_type_u8_4), sizeof(u8_4));
    REQUIRE_EQUAL_NAMED("mpi type f16_4  correct", get_mpi_size(mpi_type_f16_4), sizeof(f16_4));
    REQUIRE_EQUAL_NAMED("mpi type f32_4  correct", get_mpi_size(mpi_type_f32_4), sizeof(f32_4));
    REQUIRE_EQUAL_NAMED("mpi type f64_4  correct", get_mpi_size(mpi_type_f64_4), sizeof(f64_4));
    REQUIRE_EQUAL_NAMED("mpi type i64_8  correct", get_mpi_size(mpi_type_i64_8), sizeof(i64_8));
    REQUIRE_EQUAL_NAMED("mpi type i32_8  correct", get_mpi_size(mpi_type_i32_8), sizeof(i32_8));
    REQUIRE_EQUAL_NAMED("mpi type i16_8  correct", get_mpi_size(mpi_type_i16_8), sizeof(i16_8));
    REQUIRE_EQUAL_NAMED("mpi type i8_8   correct", get_mpi_size(mpi_type_i8_8), sizeof(i8_8));
    REQUIRE_EQUAL_NAMED("mpi type u64_8  correct", get_mpi_size(mpi_type_u64_8), sizeof(u64_8));
    REQUIRE_EQUAL_NAMED("mpi type u32_8  correct", get_mpi_size(mpi_type_u32_8), sizeof(u32_8));
    REQUIRE_EQUAL_NAMED("mpi type u16_8  correct", get_mpi_size(mpi_type_u16_8), sizeof(u16_8));
    REQUIRE_EQUAL_NAMED("mpi type u8_8   correct", get_mpi_size(mpi_type_u8_8), sizeof(u8_8));
    REQUIRE_EQUAL_NAMED("mpi type f16_8  correct", get_mpi_size(mpi_type_f16_8), sizeof(f16_8));
    REQUIRE_EQUAL_NAMED("mpi type f32_8  correct", get_mpi_size(mpi_type_f32_8), sizeof(f32_8));
    REQUIRE_EQUAL_NAMED("mpi type f64_8  correct", get_mpi_size(mpi_type_f64_8), sizeof(f64_8));
    REQUIRE_EQUAL_NAMED("mpi type i64_16 correct", get_mpi_size(mpi_type_i64_16), sizeof(i64_16));
    REQUIRE_EQUAL_NAMED("mpi type i32_16 correct", get_mpi_size(mpi_type_i32_16), sizeof(i32_16));
    REQUIRE_EQUAL_NAMED("mpi type i16_16 correct", get_mpi_size(mpi_type_i16_16), sizeof(i16_16));
    REQUIRE_EQUAL_NAMED("mpi type i8_16  correct", get_mpi_size(mpi_type_i8_16), sizeof(i8_16));
    REQUIRE_EQUAL_NAMED("mpi type u64_16 correct", get_mpi_size(mpi_type_u64_16), sizeof(u64_16));
    REQUIRE_EQUAL_NAMED("mpi type u32_16 correct", get_mpi_size(mpi_type_u32_16), sizeof(u32_16));
    REQUIRE_EQUAL_NAMED("mpi type u16_16 correct", get_mpi_size(mpi_type_u16_16), sizeof(u16_16));
    REQUIRE_EQUAL_NAMED("mpi type u8_16  correct", get_mpi_size(mpi_type_u8_16), sizeof(u8_16));
    REQUIRE_EQUAL_NAMED("mpi type f16_16 correct", get_mpi_size(mpi_type_f16_16), sizeof(f16_16));
    REQUIRE_EQUAL_NAMED("mpi type f32_16 correct", get_mpi_size(mpi_type_f32_16), sizeof(f32_16));
    REQUIRE_EQUAL_NAMED("mpi type f64_16 correct", get_mpi_size(mpi_type_f64_16), sizeof(f64_16));
}

#if false



template<class T>
void test_type1(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T(distval(eng));
    a_check[1] = T(distval(eng));

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), a_recv[0]   == a_check[0]   );
        Test_assert(format("test send/recv %s[1]  ",name), a_recv[1]   == a_check[1]   );

    }

}


template<class T>
void test_type2(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T{distval(eng),distval(eng)};
    a_check[1] = T{distval(eng),distval(eng)};

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), test_sycl_eq(a_recv[0] , a_check[0])   );
        Test_assert(format("test send/recv %s[1]  ",name), test_sycl_eq(a_recv[1] , a_check[1])   );

    }

}


template<class T>
void test_type3(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T{distval(eng),distval(eng),distval(eng)};
    a_check[1] = T{distval(eng),distval(eng),distval(eng)};

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), test_sycl_eq(a_recv[0] , a_check[0])   );
        Test_assert(format("test send/recv %s[1]  ",name), test_sycl_eq(a_recv[1] , a_check[1])   );

    }

}


template<class T>
void test_type4(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T{distval(eng),distval(eng),distval(eng),distval(eng)};
    a_check[1] = T{distval(eng),distval(eng),distval(eng),distval(eng)};

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), test_sycl_eq(a_recv[0] , a_check[0])   );
        Test_assert(format("test send/recv %s[1]  ",name), test_sycl_eq(a_recv[1] , a_check[1])   );

    }

}


template<class T>
void test_type8(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T{distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng)};
    a_check[1] = T{distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng)};

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), test_sycl_eq(a_recv[0] , a_check[0])   );
        Test_assert(format("test send/recv %s[1]  ",name), test_sycl_eq(a_recv[1] , a_check[1])   );

    }

}


template<class T>
void test_type16(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){

    T a_check [2];
    a_check[0] = T{distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng)};
    a_check[1] = T{distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng),distval(eng)};

    if(shamcomm::world_rank() == 0){
        mpi::send(&a_check   , 2, mpi_t   , 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){

        T a_recv [2];
        MPI_Status st;
        mpi::recv(&a_recv   , 2, mpi_t   , 0, 0, MPI_COMM_WORLD, &st);

        Test_assert(format("test send/recv %s[0]  ",name), test_sycl_eq(a_recv[0] , a_check[0])   );
        Test_assert(format("test send/recv %s[1]  ",name), test_sycl_eq(a_recv[1] , a_check[1])   );

    }

}


Test_start("sycl_handler::",test_sycl_mpi_types_comm,2){

    std::mt19937 eng(0x1171);
    std::uniform_real_distribution<f64> distval(-1e9,1e9);


    test_type1<i64>(__test_result_ref, eng, distval,mpi_type_i64,"i64");
    test_type1<i32>(__test_result_ref, eng, distval,mpi_type_i32,"i32");
    test_type1<i16>(__test_result_ref, eng, distval,mpi_type_i16,"i16");
    test_type1<i8 >(__test_result_ref, eng, distval,mpi_type_i8 ,"i8 ");
    test_type1<u64>(__test_result_ref, eng, distval,mpi_type_u64,"u64");
    test_type1<u32>(__test_result_ref, eng, distval,mpi_type_u32,"u32");
    test_type1<u16>(__test_result_ref, eng, distval,mpi_type_u16,"u16");
    test_type1<u8 >(__test_result_ref, eng, distval,mpi_type_u8 ,"u8 ");
    test_type1<f16>(__test_result_ref, eng, distval,mpi_type_f16,"f16");
    test_type1<f32>(__test_result_ref, eng, distval,mpi_type_f32,"f32");
    test_type1<f64>(__test_result_ref, eng, distval,mpi_type_f64,"f64");

    test_type2<i64_2>(__test_result_ref, eng, distval,mpi_type_i64_2,"i64_2");
    test_type2<i32_2>(__test_result_ref, eng, distval,mpi_type_i32_2,"i32_2");
    test_type2<i16_2>(__test_result_ref, eng, distval,mpi_type_i16_2,"i16_2");
    test_type2<i8_2 >(__test_result_ref, eng, distval,mpi_type_i8_2 ,"i8_2 ");
    test_type2<u64_2>(__test_result_ref, eng, distval,mpi_type_u64_2,"u64_2");
    test_type2<u32_2>(__test_result_ref, eng, distval,mpi_type_u32_2,"u32_2");
    test_type2<u16_2>(__test_result_ref, eng, distval,mpi_type_u16_2,"u16_2");
    test_type2<u8_2 >(__test_result_ref, eng, distval,mpi_type_u8_2 ,"u8_2 ");
    test_type2<f16_2>(__test_result_ref, eng, distval,mpi_type_f16_2,"f16_2");
    test_type2<f32_2>(__test_result_ref, eng, distval,mpi_type_f32_2,"f32_2");
    test_type2<f64_2>(__test_result_ref, eng, distval,mpi_type_f64_2,"f64_2");

    test_type3<i64_3>(__test_result_ref, eng, distval,mpi_type_i64_3,"i64_3");
    test_type3<i32_3>(__test_result_ref, eng, distval,mpi_type_i32_3,"i32_3");
    test_type3<i16_3>(__test_result_ref, eng, distval,mpi_type_i16_3,"i16_3");
    test_type3<i8_3 >(__test_result_ref, eng, distval,mpi_type_i8_3 ,"i8_3 ");
    test_type3<u64_3>(__test_result_ref, eng, distval,mpi_type_u64_3,"u64_3");
    test_type3<u32_3>(__test_result_ref, eng, distval,mpi_type_u32_3,"u32_3");
    test_type3<u16_3>(__test_result_ref, eng, distval,mpi_type_u16_3,"u16_3");
    test_type3<u8_3 >(__test_result_ref, eng, distval,mpi_type_u8_3 ,"u8_3 ");
    test_type3<f16_3>(__test_result_ref, eng, distval,mpi_type_f16_3,"f16_3");
    test_type3<f32_3>(__test_result_ref, eng, distval,mpi_type_f32_3,"f32_3");
    test_type3<f64_3>(__test_result_ref, eng, distval,mpi_type_f64_3,"f64_3");

    test_type4<i64_4>(__test_result_ref, eng, distval,mpi_type_i64_4,"i64_4");
    test_type4<i32_4>(__test_result_ref, eng, distval,mpi_type_i32_4,"i32_4");
    test_type4<i16_4>(__test_result_ref, eng, distval,mpi_type_i16_4,"i16_4");
    test_type4<i8_4 >(__test_result_ref, eng, distval,mpi_type_i8_4 ,"i8_4 ");
    test_type4<u64_4>(__test_result_ref, eng, distval,mpi_type_u64_4,"u64_4");
    test_type4<u32_4>(__test_result_ref, eng, distval,mpi_type_u32_4,"u32_4");
    test_type4<u16_4>(__test_result_ref, eng, distval,mpi_type_u16_4,"u16_4");
    test_type4<u8_4 >(__test_result_ref, eng, distval,mpi_type_u8_4 ,"u8_4 ");
    test_type4<f16_4>(__test_result_ref, eng, distval,mpi_type_f16_4,"f16_4");
    test_type4<f32_4>(__test_result_ref, eng, distval,mpi_type_f32_4,"f32_4");
    test_type4<f64_4>(__test_result_ref, eng, distval,mpi_type_f64_4,"f64_4");

    test_type8<i64_8>(__test_result_ref, eng, distval,mpi_type_i64_8,"i64_8");
    test_type8<i32_8>(__test_result_ref, eng, distval,mpi_type_i32_8,"i32_8");
    test_type8<i16_8>(__test_result_ref, eng, distval,mpi_type_i16_8,"i16_8");
    test_type8<i8_8 >(__test_result_ref, eng, distval,mpi_type_i8_8 ,"i8_8 ");
    test_type8<u64_8>(__test_result_ref, eng, distval,mpi_type_u64_8,"u64_8");
    test_type8<u32_8>(__test_result_ref, eng, distval,mpi_type_u32_8,"u32_8");
    test_type8<u16_8>(__test_result_ref, eng, distval,mpi_type_u16_8,"u16_8");
    test_type8<u8_8 >(__test_result_ref, eng, distval,mpi_type_u8_8 ,"u8_8 ");
    test_type8<f16_8>(__test_result_ref, eng, distval,mpi_type_f16_8,"f16_8");
    test_type8<f32_8>(__test_result_ref, eng, distval,mpi_type_f32_8,"f32_8");
    test_type8<f64_8>(__test_result_ref, eng, distval,mpi_type_f64_8,"f64_8");

    test_type16<i64_16>(__test_result_ref, eng, distval,mpi_type_i64_16,"i64_16");
    test_type16<i32_16>(__test_result_ref, eng, distval,mpi_type_i32_16,"i32_16");
    test_type16<i16_16>(__test_result_ref, eng, distval,mpi_type_i16_16,"i16_16");
    test_type16<i8_16 >(__test_result_ref, eng, distval,mpi_type_i8_16 ,"i8_16 ");
    test_type16<u64_16>(__test_result_ref, eng, distval,mpi_type_u64_16,"u64_16");
    test_type16<u32_16>(__test_result_ref, eng, distval,mpi_type_u32_16,"u32_16");
    test_type16<u16_16>(__test_result_ref, eng, distval,mpi_type_u16_16,"u16_16");
    test_type16<u8_16 >(__test_result_ref, eng, distval,mpi_type_u8_16 ,"u8_16 ");
    test_type16<f16_16>(__test_result_ref, eng, distval,mpi_type_f16_16,"f16_16");
    test_type16<f32_16>(__test_result_ref, eng, distval,mpi_type_f32_16,"f32_16");
    test_type16<f64_16>(__test_result_ref, eng, distval,mpi_type_f64_16,"f64_16");



}




template<class T> inline void test_type_comm(TestResults &__test_result_ref,std::mt19937 &  eng, std::uniform_real_distribution<f64> & distval,MPI_Datatype mpi_t,const char* name){
    constexpr u32 npart = 1000;

    using namespace mpi_sycl_interop;

    static_assert(
    #define X(args) std::is_same<T, args>::value ||
            XMAC_SYCLMPI_TYPE_ENABLED true
    #undef X
            );

    std::unique_ptr<sycl::buffer<T>> buf = std::make_unique<sycl::buffer<T>>(npart);

    {
        sycl::host_accessor acc {*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < npart; i++) {
            acc[i] = next_obj<T>(eng, distval);
        }
    }

    std::vector<mpi_sycl_interop::BufferMpiRequest<T>> rqs;

    if(shamcomm::world_rank() == 0){
        isend(buf,npart, rqs, 1, 0, MPI_COMM_WORLD);

        waitall(rqs);
    }else{
        std::unique_ptr<sycl::buffer<T>> recv;
        irecv_probe(recv, rqs, 0, 0, MPI_COMM_WORLD);


        waitall(rqs);


        {
            sycl::host_accessor acc {*buf, sycl::read_only};
            sycl::host_accessor acc_recv {*recv, sycl::read_only};

            for (u32 i = 0; i < npart; i++) {
                Test_assert("equal", test_sycl_eq(acc[i],acc_recv[i]));
            }
        }


    }

}



Test_start("sycl_mpi_interop::",test_sycl_buffer_mpi_comm,2){



    std::mt19937 eng(0x1171);
    std::uniform_real_distribution<f64> distval(-1e9,1e9);

    #define X(args) test_type_comm<args>(__test_result_ref, eng, distval, mpi_type_##args, #args);
    XMAC_SYCLMPI_TYPE_ENABLED
    #undef X


}

#endif
