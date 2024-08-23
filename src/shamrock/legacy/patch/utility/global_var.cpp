// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file global_var.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "global_var.hpp"
#include <cmath>

template<class T, GlobalVariableType redop>
T int_reduce_get_start_var();

template<>
f32 int_reduce_get_start_var<f32, GlobalVariableType::min>() {
    return INFINITY;
}

template<>
f64 int_reduce_get_start_var<f64, GlobalVariableType::min>() {
    return INFINITY;
}

template<>
f32 int_reduce_get_start_var<f32, GlobalVariableType::max>() {
    return 0;
}

template<>
f64 int_reduce_get_start_var<f64, GlobalVariableType::max>() {
    return 0;
}

template<>
f32 int_reduce_get_start_var<f32, GlobalVariableType::sum>() {
    return 0;
}

template<>
f64 int_reduce_get_start_var<f64, GlobalVariableType::sum>() {
    return 0;
}

template<class T, GlobalVariableType redop>
T int_reduce_val_loc(T a, T b);

template<>
f32 int_reduce_val_loc<f32, GlobalVariableType::min>(f32 a, f32 b) {
    return sycl::min(a, b);
}

template<>
f32 int_reduce_val_loc<f32, GlobalVariableType::max>(f32 a, f32 b) {
    return sycl::max(a, b);
}

template<>
f32 int_reduce_val_loc<f32, GlobalVariableType::sum>(f32 a, f32 b) {
    return a + b;
}

template<class T, GlobalVariableType redop>
T int_reduce_val_mpi(T val_acc_loc);

template<>
f32 int_reduce_val_mpi<f32, GlobalVariableType::min>(f32 val_acc_loc) {
    f32 ret;
    mpi::allreduce(&val_acc_loc, &ret, 1, mpi_type_f32, MPI_MIN, MPI_COMM_WORLD);
    return ret;
}

template<>
f32 int_reduce_val_mpi<f32, GlobalVariableType::max>(f32 val_acc_loc) {
    f32 ret;
    mpi::allreduce(&val_acc_loc, &ret, 1, mpi_type_f32, MPI_MAX, MPI_COMM_WORLD);
    return ret;
}

template<>
f32 int_reduce_val_mpi<f32, GlobalVariableType::sum>(f32 val_acc_loc) {
    f32 ret;
    mpi::allreduce(&val_acc_loc, &ret, 1, mpi_type_f32, MPI_SUM, MPI_COMM_WORLD);
    return ret;
}

template<class T, GlobalVariableType redop>
T int_reduce(std::unordered_map<u64, T> &val_map) {
    T val_acc_loc = int_reduce_get_start_var<T, redop>();
    for (auto &[k, val] : val_map) {
        val_acc_loc = int_reduce_val_loc<T, redop>(val_acc_loc, val);
    }

    return int_reduce_val_mpi<T, redop>(val_acc_loc);
}

template<>
void GlobalVariable<GlobalVariableType::min, f32>::reduce_val() {

    final_val = int_reduce<f32, GlobalVariableType::min>(val_map);

    is_reduced = true;
}

template<>
void GlobalVariable<GlobalVariableType::max, f32>::reduce_val() {

    final_val = int_reduce<f32, GlobalVariableType::max>(val_map);

    is_reduced = true;
}

template<>
void GlobalVariable<GlobalVariableType::sum, f32>::reduce_val() {

    final_val = int_reduce<f32, GlobalVariableType::sum>(val_map);

    is_reduced = true;
}
