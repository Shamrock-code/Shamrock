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
 * @file exchanges.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/wrapper.hpp"
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
     */
    template<class T>
    inline void vector_allgatherv(
        const std::vector<T> &send_vec,
        const MPI_Datatype &send_type,
        std::vector<T> &recv_vec,
        const MPI_Datatype &recv_type,
        const MPI_Comm comm) {
        StackEntry stack_loc{};

        u32 local_count = send_vec.size();

        std::vector<int> table_data_count(shamcomm::world_size());

        shamcomm::mpi::Allgather(&local_count, 1, MPI_INT, &table_data_count[0], 1, MPI_INT, comm);

        std::vector<int> node_displacments_data_table(shamcomm::world_size());
        node_displacments_data_table[0] = 0;
        for (u32 i = 1; i < shamcomm::world_size(); i++) {
            node_displacments_data_table[i]
                = node_displacments_data_table[i - 1] + table_data_count[i - 1];
        }

        // use work duplication or MPI reduction
        {
#if false
            // querry global size and resize the receiving vector
            u32 global_len;
            shamcomm::mpi::Allreduce(
                &local_count, &global_len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            recv_vec.resize(global_len);
#else
            u32 global_len = 0;
            for (const u32 &v : table_data_count) {
                global_len += v;
            }
            recv_vec.resize(global_len);
#endif
        }

        auto send_ptr = (send_vec.size() > 0) ? &send_vec[0] : nullptr;
        auto recv_ptr = (recv_vec.size() > 0) ? &recv_vec[0] : nullptr;

        shamcomm::mpi::Allgatherv(
            send_ptr,
            send_vec.size(),
            send_type,
            recv_ptr,
            &table_data_count[0],
            &node_displacments_data_table[0],
            recv_type,
            comm);
    }

    template<class T>
    inline void vector_allgatherv(
        const std::vector<T> &send_vec, std::vector<T> &recv_vec, const MPI_Comm comm) {
        vector_allgatherv(send_vec, get_mpi_type<T>(), recv_vec, get_mpi_type<T>(), comm);
    }

} // namespace shamalgs::collective
