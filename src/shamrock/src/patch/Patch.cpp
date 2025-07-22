// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Patch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/patch/Patch.hpp"
#include "shamsys/MpiDataTypeHandler.hpp"

namespace shamrock::patch {

    MPI_Datatype patch_3d_MPI_type;

    template<>
    MPI_Datatype get_patch_mpi_type<3>() {
        return patch_3d_MPI_type;
    }
} // namespace shamrock::patch

/////////////////////////////////////////////
// MPI init related to patches
/////////////////////////////////////////////

MPI_Datatype patch_3d_MPI_types_list[2];
int patch_3d_MPI_block_lens[2];
MPI_Aint patch_3d_MPI_offset[2];

Register_MPIDtypeInit(init_patch_type, "mpi patch type") {
    using namespace shamrock::patch;

    patch_3d_MPI_block_lens[0] = 9; // 9 u64
    patch_3d_MPI_block_lens[1] = 1; // 2 u32

    patch_3d_MPI_types_list[0] = MPI_LONG;
    patch_3d_MPI_types_list[1] = MPI_INT;

    patch_3d_MPI_offset[0] = offsetof(shamrock::patch::Patch, id_patch);
    patch_3d_MPI_offset[1] = offsetof(shamrock::patch::Patch, node_owner_id);

    mpi::type_create_struct(
        2,
        patch_3d_MPI_block_lens,
        patch_3d_MPI_offset,
        patch_3d_MPI_types_list,
        &patch_3d_MPI_type);
    mpi::type_commit(&patch_3d_MPI_type);
}

Register_MPIDtypeFree(free_patch_type, "mpi patch type") {
    using namespace shamrock::patch;

    mpi::type_free(&patch_3d_MPI_type);
}
