// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file exchanges.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/are_all_rank_true.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
#include <numeric>
#include <vector>

namespace shamalgs::collective {

    /**
     * @brief allgatherv with knowing total count of object
     * //TODO add fault tolerance
     * @tparam T
     * @param send_vec
     * @param send_type
     * @param recv_vec
     * @param recv_type
     */
    template<class T>
    inline void vector_allgatherv_ks(
        const std::vector<T> &send_vec,
        const MPI_Datatype send_type,
        std::vector<T> &recv_vec,
        const MPI_Datatype recv_type,
        const MPI_Comm comm) {

        u32 local_count = send_vec.size();

        int *table_data_count = new int[shamcomm::world_size()];

        // crash
        shamcomm::mpi::Allgather(&local_count, 1, MPI_INT, &table_data_count[0], 1, MPI_INT, comm);

        // printf("table_data_count =
        // [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);

        int *node_displacments_data_table = new int[shamcomm::world_size()];

        node_displacments_data_table[0] = 0;

        for (u32 i = 1; i < shamcomm::world_size(); i++) {
            node_displacments_data_table[i]
                = node_displacments_data_table[i - 1] + table_data_count[i - 1];
        }

        // printf("node_displacments_data_table =
        // [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);

        shamcomm::mpi::Allgatherv(
            &send_vec[0],
            send_vec.size(),
            send_type,
            &recv_vec[0],
            table_data_count,
            node_displacments_data_table,
            recv_type,
            comm);

        delete[] table_data_count;
        delete[] node_displacments_data_table;
    }

    /**
     * @brief allgatherv on vector with size query (size querrying variant of vector_allgatherv_ks)
     * //TODO add fault tolerance
     * @tparam T
     * @param send_vec
     * @param send_type
     * @param recv_vec
     * @param recv_type
     * @return the node displacments data table
     */
    template<class T>
    inline std::vector<int> vector_allgatherv(
        const std::vector<T> &send_vec,
        const MPI_Datatype &send_type,
        std::vector<T> &recv_vec,
        const MPI_Datatype &recv_type,
        const MPI_Comm comm) {
        StackEntry stack_loc{};

        int comm_size = 0;

        if (comm == MPI_COMM_WORLD) {
            comm_size = shamcomm::world_size();
        } else {
            MPICHECK(MPI_Comm_size(comm, &comm_size));
        }

        int local_count = shambase::narrow_or_throw<int>(send_vec.size());

        std::vector<int> table_data_count(static_cast<std::size_t>(comm_size));

        shamcomm::mpi::Allgather(
            &local_count, 1, MPI_INT, table_data_count.data(), 1, MPI_INT, comm);

        int global_len = 0;
        // use work duplication or MPI reduction
#if false
        // querry global size and resize the receiving vector
        shamcomm::mpi::Allreduce(
            &local_count, &global_len, 1, MPI_INT, MPI_SUM, comm);
#else
        {
            u64 tmp = std::accumulate(table_data_count.begin(), table_data_count.end(), 0_u64);

            // if it exceeds the max size of int, MPI will trip like crazy
            // god damn it just implement 64bits indicies ... Pleeeeeasssssse !!!
            global_len = shambase::narrow_or_throw<int>(tmp);
        }
#endif

        recv_vec.resize(global_len);

        if (global_len == 0) {
            return {};
        }

        // here we can not overflow since we know that the sum can be narrowed to an int
        std::vector<int> node_displacments_data_table(static_cast<std::size_t>(comm_size));
        std::exclusive_scan(
            table_data_count.begin(),
            table_data_count.end(),
            node_displacments_data_table.begin(),
            0);

        shamcomm::mpi::Allgatherv(
            send_vec.data(), // even if the size is 0 MPI does not care
            local_count,
            send_type,
            recv_vec.data(),
            table_data_count.data(),
            node_displacments_data_table.data(),
            recv_type,
            comm);

        return node_displacments_data_table;
    }

    /**
     * @brief vector_allgatherv version that support having more than 2^31 elements in flight
     * @tparam T
     * @param send_vec
     * @param send_type
     * @param recv_vec
     * @param recv_type
     * @param comm
     * @param com_per_step
     */
    template<class T>
    inline void vector_allgatherv_large(
        const std::vector<T> &send_vec,
        const MPI_Datatype &send_type,
        std::vector<T> &recv_vec,
        const MPI_Datatype &recv_type,
        const MPI_Comm comm,
        u32 com_per_step = (1_i32 << 29) / static_cast<u32>(shamcomm::world_size())) {

        // check that comm is MPI_COMM_WORLD
        if (comm != MPI_COMM_WORLD) {
            throw shambase::make_except_with_loc<std::runtime_error>("comm must be MPI_COMM_WORLD");
        }

        u64 send_offset = 0_u64;
        std::vector<u64> result_disps(shamcomm::world_size() + 1, 0_u64);

        while (!shamalgs::collective::are_all_rank_true(send_offset == send_vec.size(), comm)) {
            // extract com_per_step elements from send_vec
            u64 remaining
                = (send_offset < send_vec.size()) ? (send_vec.size() - send_offset) : 0_u64;
            u64 num_to_send = std::min<u64>(com_per_step, remaining);
            std::vector<T> send_vec_internal(
                send_vec.begin() + send_offset, send_vec.begin() + send_offset + num_to_send);
            send_offset += num_to_send;

            std::vector<T> recv_vec_internal{};
            auto disp = vector_allgatherv(
                send_vec_internal, send_type, recv_vec_internal, recv_type, comm);
            disp.push_back(shambase::narrow_or_throw<int>(recv_vec_internal.size()));

            // The bit that insert in such a way that it reproduce vector_allgatherv
            for (u32 i = 0; i < (disp.size() - 1); i++) {
                auto insert_loc = recv_vec.begin() + result_disps[i + 1] + disp[i];
                recv_vec.insert(
                    insert_loc,
                    recv_vec_internal.begin() + disp[i],
                    recv_vec_internal.begin() + disp[i + 1]);
                result_disps[i] += disp[i];
            }
            result_disps[disp.size() - 1] += disp[disp.size() - 1];
        }
    }

    /**
     * @brief Simplified allgatherv wrapper using default MPI type
     *
     * @tparam T Type of elements to gather
     * @param send_vec Vector to send from this rank
     * @param recv_vec Vector to receive all gathered data
     * @param comm MPI communicator
     */
    template<class T>
    inline void vector_allgatherv(
        const std::vector<T> &send_vec, std::vector<T> &recv_vec, const MPI_Comm comm) {
        vector_allgatherv(send_vec, get_mpi_type<T>(), recv_vec, get_mpi_type<T>(), comm);
    }

} // namespace shamalgs::collective
