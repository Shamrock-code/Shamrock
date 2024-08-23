// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file collectives.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"

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
