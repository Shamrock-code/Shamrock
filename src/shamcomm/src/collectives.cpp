// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file collectives.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include <unordered_map>

void shamcomm::gather_str(const std::string &send_vec, std::string &recv_vec) {
    StackEntry stack_loc{};

    u32 local_count = send_vec.size();

    // querry global size and resize the receiving vector
    u32 global_len;
    MPICHECK(MPI_Allreduce(&local_count, &global_len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    recv_vec.resize(global_len);

    int *table_data_count = new int[shamcomm::world_size()];

    MPICHECK(
        MPI_Allgather(&local_count, 1, MPI_INT, &table_data_count[0], 1, MPI_INT, MPI_COMM_WORLD));

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

    MPICHECK(MPI_Allgatherv(
        send_vec.data(),
        send_vec.size(),
        MPI_CHAR,
        recv_vec.data(),
        table_data_count,
        node_displacments_data_table,
        MPI_CHAR,
        MPI_COMM_WORLD));

    delete[] table_data_count;
    delete[] node_displacments_data_table;
}

std::unordered_map<std::string, int>
shamcomm::string_histogram(const std::vector<std::string> &inputs, std::string delimiter) {
    std::string accum_loc = "";
    for (auto &s : inputs) {
        accum_loc += s + delimiter;
    }

    std::string recv = "";
    gather_str(accum_loc, recv);

    if (world_rank() == 0) {

        std::vector<std::string> splitted = shambase::split_str(recv, delimiter);

        std::unordered_map<std::string, int> histogram;

        for (size_t i = 0; i < splitted.size(); i++) {
            histogram[splitted[i]] += 1;
        }

        return histogram;
    }

    return {};
}
