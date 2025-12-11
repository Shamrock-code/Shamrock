// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/narrowing.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"

#define XMAC_SYCLMPI_TYPE_TO_TEST                                                                  \
    X(i64)                                                                                         \
    X(i32)                                                                                         \
    X(i16)                                                                                         \
    X(i8)                                                                                          \
    X(u64)                                                                                         \
    X(u32)                                                                                         \
    X(u16)                                                                                         \
    X(u8)                                                                                          \
    X(f32)                                                                                         \
    X(f64)                                                                                         \
    X(i64_2)                                                                                       \
    X(i32_2)                                                                                       \
    X(i16_2)                                                                                       \
    X(i8_2)                                                                                        \
    X(u64_2)                                                                                       \
    X(u32_2)                                                                                       \
    X(u16_2)                                                                                       \
    X(u8_2)                                                                                        \
    X(f32_2)                                                                                       \
    X(f64_2)                                                                                       \
    X(i64_3)                                                                                       \
    X(i32_3)                                                                                       \
    X(i16_3)                                                                                       \
    X(i8_3)                                                                                        \
    X(u64_3)                                                                                       \
    X(u32_3)                                                                                       \
    X(u16_3)                                                                                       \
    X(u8_3)                                                                                        \
    X(f32_3)                                                                                       \
    X(f64_3)                                                                                       \
    X(i64_4)                                                                                       \
    X(i32_4)                                                                                       \
    X(i16_4)                                                                                       \
    X(i8_4)                                                                                        \
    X(u64_4)                                                                                       \
    X(u32_4)                                                                                       \
    X(u16_4)                                                                                       \
    X(u8_4)                                                                                        \
    X(f32_4)                                                                                       \
    X(f64_4)                                                                                       \
    X(i64_8)                                                                                       \
    X(i32_8)                                                                                       \
    X(i16_8)                                                                                       \
    X(i8_8)                                                                                        \
    X(u64_8)                                                                                       \
    X(u32_8)                                                                                       \
    X(u16_8)                                                                                       \
    X(u8_8)                                                                                        \
    X(f32_8)                                                                                       \
    X(f64_8)                                                                                       \
    X(i64_16)                                                                                      \
    X(i32_16)                                                                                      \
    X(i16_16)                                                                                      \
    X(i8_16)                                                                                       \
    X(u64_16)                                                                                      \
    X(u32_16)                                                                                      \
    X(u16_16)                                                                                      \
    X(u8_16)                                                                                       \
    X(f32_16)                                                                                      \
    X(f64_16)

size_t get_mpi_size(MPI_Datatype md) {
    MPI_Aint lbs;
    MPI_Aint exts;

    MPICHECK(MPI_Type_get_extent(md, &lbs, &exts));
    return shambase::narrow_or_throw<size_t>(exts);
}

TestStart(Unittest, "shambackends/test_sycl_mpi_types_sizes", test_sycl_mpi_types_sizes, -1) {

#define X(args)                                                                                    \
    REQUIRE_EQUAL_NAMED("mpi type " #args " correct", get_mpi_size(mpi_type_##args), sizeof(args));
    XMAC_SYCLMPI_TYPE_TO_TEST
#undef X
}

template<class T>
void test_comm() {
    // single element communication
    std::mt19937 eng(0x111);

    auto get_bounds = []() {
        if constexpr (std::is_same_v<shambase::VecComponent<T>, f16>) {
            return std::make_pair(T{1.0f}, T{1.0f});
        } else {
            return std::make_pair(
                shambase::VectorProperties<T>::get_min(), shambase::VectorProperties<T>::get_max());
        }
    };

    auto [bound_min, bound_max] = get_bounds();

    T val_test = shamalgs::primitives::mock_value<T>(eng, bound_min, bound_max);

    if (shamcomm::world_rank() == 0) {
        MPICHECK(MPI_Send(&val_test, 1, get_mpi_type<T>(), 1, 0, MPI_COMM_WORLD));
    }
    if (shamcomm::world_rank() == 1) {
        T val_recv;
        MPICHECK(
            MPI_Recv(&val_recv, 1, get_mpi_type<T>(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        REQUIRE_EQUAL_CUSTOM_COMP(val_test, val_recv, sham::equals);
    }

    // array communication
    std::vector<T> val_test_array
        = shamalgs::primitives::mock_vector<T>(0x111, 420, bound_min, bound_max);

    if (shamcomm::world_rank() == 0) {
        MPICHECK(MPI_Send(
            val_test_array.data(), val_test_array.size(), get_mpi_type<T>(), 1, 0, MPI_COMM_WORLD));
    }
    if (shamcomm::world_rank() == 1) {
        std::vector<T> val_recv_array(val_test_array.size());
        MPICHECK(MPI_Recv(
            val_recv_array.data(),
            val_test_array.size(),
            get_mpi_type<T>(),
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE));

        auto vec_equals = [](std::vector<T> a, std::vector<T> b) {
            bool same_size = a.size() == b.size();
            if (!same_size) {
                return false;
            }
            for (size_t i = 0; i < a.size(); i++) {
                same_size = same_size && sham::equals(a[i], b[i]);
            }
            return same_size;
        };
        REQUIRE_EQUAL_CUSTOM_COMP(val_test_array, val_recv_array, vec_equals);
    }
}

TestStart(Unittest, "shambackends/test_sycl_mpi_types_comm", test_sycl_mpi_types_comm, 2) {

#define X(args) test_comm<args>();
    XMAC_SYCLMPI_TYPE_TO_TEST
#undef X
}
