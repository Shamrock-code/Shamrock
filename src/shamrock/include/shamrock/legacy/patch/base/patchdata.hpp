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
 * @file patchdata.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header for PatchData related function and declaration
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "patchdata_field.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <random>
#include <variant>
#include <vector>

struct PatchDataMpiRequest {
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32>> mpi_rq_fields_f32;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_2>> mpi_rq_fields_f32_2;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_3>> mpi_rq_fields_f32_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_4>> mpi_rq_fields_f32_4;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_8>> mpi_rq_fields_f32_8;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f32_16>> mpi_rq_fields_f32_16;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64>> mpi_rq_fields_f64;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_2>> mpi_rq_fields_f64_2;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_3>> mpi_rq_fields_f64_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_4>> mpi_rq_fields_f64_4;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_8>> mpi_rq_fields_f64_8;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<f64_16>> mpi_rq_fields_f64_16;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u32>> mpi_rq_fields_u32;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u64>> mpi_rq_fields_u64;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u32_3>> mpi_rq_fields_u32_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<u64_3>> mpi_rq_fields_u64_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<i64_3>> mpi_rq_fields_i64_3;
    std::vector<patchdata_field::PatchDataFieldMpiRequest<i64>> mpi_rq_fields_i64;

    inline void finalize() {
        for (auto b : mpi_rq_fields_f32) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f32_2) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f32_3) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f32_4) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f32_8) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f32_16) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64_2) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64_3) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64_4) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64_8) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_f64_16) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_u32) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_u64) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_u32_3) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_u64_3) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_i64_3) {
            b.finalize();
        }
        for (auto b : mpi_rq_fields_i64) {
            b.finalize();
        }
    }

    template<class T>
    std::vector<patchdata_field::PatchDataFieldMpiRequest<T>> &get_field_list();
#define X(_arg)                                                                                    \
    template<>                                                                                     \
    inline std::vector<patchdata_field::PatchDataFieldMpiRequest<_arg>> &get_field_list() {        \
        return mpi_rq_fields_##_arg;                                                               \
    }
    XMAC_LIST_ENABLED_FIELD
#undef X
};

inline void waitall_pdat_mpi_rq(std::vector<PatchDataMpiRequest> &rq_lst) {

    std::vector<MPI_Request> rqst;

    auto insertor = [&](auto in) {
        std::vector<MPI_Request> rloc = patchdata_field::get_rqs(in);
        rqst.insert(rqst.end(), rloc.begin(), rloc.end());
    };

    for (auto a : rq_lst) {
        insertor(a.mpi_rq_fields_f32);
        insertor(a.mpi_rq_fields_f32_2);
        insertor(a.mpi_rq_fields_f32_3);
        insertor(a.mpi_rq_fields_f32_4);
        insertor(a.mpi_rq_fields_f32_8);
        insertor(a.mpi_rq_fields_f32_16);
        insertor(a.mpi_rq_fields_f64);
        insertor(a.mpi_rq_fields_f64_2);
        insertor(a.mpi_rq_fields_f64_3);
        insertor(a.mpi_rq_fields_f64_4);
        insertor(a.mpi_rq_fields_f64_8);
        insertor(a.mpi_rq_fields_f64_16);
        insertor(a.mpi_rq_fields_u32);
        insertor(a.mpi_rq_fields_u64);
        insertor(a.mpi_rq_fields_u32_3);
        insertor(a.mpi_rq_fields_u64_3);
        insertor(a.mpi_rq_fields_i64_3);
    }

    std::vector<MPI_Status> st_lst(rqst.size());
    shamcomm::mpi::Waitall(rqst.size(), rqst.data(), st_lst.data());

    for (auto a : rq_lst) {
        a.finalize();
    }
}

/**
 * @brief perform a MPI isend with a PatchData object
 *
 * @param p the patchdata to send
 * @param rq_lst reference to the vector of MPI_Request corresponding to the send
 * @param rank_dest rabk to send data to
 * @param tag MPI communication tag
 * @param comm MPI communicator
 */
[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
u64 patchdata_isend(
    shamrock::patch::PatchData &p,
    std::vector<PatchDataMpiRequest> &rq_lst,
    i32 rank_dest,
    i32 tag,
    MPI_Comm comm);

/**
 * @brief perform a MPI irecv with a PatchData object
 *
 * @param rq_lst reference to the vector of MPI_Request corresponding to the recv
 * @param rank_source rank to receive from
 * @param tag MPI communication tag
 * @param comm  MPI communicator
 * @return the received patchdata (it works but weird because asynchronous)
 */
[[deprecated("Please use CommunicationBuffer & SerializeHelper instead")]]
u64 patchdata_irecv_probe(
    shamrock::patch::PatchData &pdat,
    std::vector<PatchDataMpiRequest> &rq_lst,
    i32 rank_source,
    i32 tag,
    MPI_Comm comm);

/**
 * @brief generate dummy patchdata from a mersen twister
 *
 * @param eng the mersen twister
 * @return PatchData the generated PatchData
 */
shamrock::patch::PatchData
patchdata_gen_dummy_data(shamrock::patch::PatchDataLayout &pdl, std::mt19937 &eng);

/**
 * @brief check if two PatchData content match
 *
 * @param p1
 * @param p2
 * @return true
 * @return false
 */
bool patch_data_check_match(shamrock::patch::PatchData &p1, shamrock::patch::PatchData &p2);
