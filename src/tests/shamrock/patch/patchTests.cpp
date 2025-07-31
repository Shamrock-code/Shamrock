// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/wrapper.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/Patch.cpp:MpiType", patch_mpi_type, 2) {

    using namespace shamrock::patch;

    Patch check_patch{};
    check_patch.id_patch        = 156;
    check_patch.pack_node_index = 48414;
    check_patch.load_value      = 4951956;
    check_patch.coord_min[0]    = 0;
    check_patch.coord_min[1]    = 1;
    check_patch.coord_min[2]    = 2;
    check_patch.coord_max[0]    = 3;
    check_patch.coord_max[1]    = 8;
    check_patch.coord_max[2]    = 6;
    check_patch.node_owner_id   = 44444;

    if (shamcomm::world_rank() == 0) {
        shamcomm::mpi::Send(&check_patch, 1, get_patch_mpi_type<3>(), 1, 0, MPI_COMM_WORLD);
    }

    if (shamcomm::world_rank() == 1) {
        Patch rpatch{};

        MPI_Status st;
        shamcomm::mpi::Recv(&rpatch, 1, get_patch_mpi_type<3>(), 0, 0, MPI_COMM_WORLD, &st);

        REQUIRE_NAMED("patch are equal", rpatch == check_patch);
    }
}

TestStart(Unittest, "shamrock/patch/Patch.cpp:SplitMerge", splitmergepatch, 1) {

    using namespace shamrock::patch;

    Patch check_patch{};
    check_patch.id_patch        = 0;
    check_patch.pack_node_index = u64_max;
    check_patch.load_value      = 8;
    check_patch.coord_min[0]    = 0;
    check_patch.coord_min[1]    = 0;
    check_patch.coord_min[2]    = 0;
    check_patch.coord_max[0]    = 256;
    check_patch.coord_max[1]    = 128;
    check_patch.coord_max[2]    = 1024;
    check_patch.node_owner_id   = 0;

    std::array<Patch, 8> splts = check_patch.get_split();

    REQUIRE(splts[0].load_value == 1);
    REQUIRE(splts[1].load_value == 1);
    REQUIRE(splts[2].load_value == 1);
    REQUIRE(splts[3].load_value == 1);
    REQUIRE(splts[4].load_value == 1);
    REQUIRE(splts[5].load_value == 1);
    REQUIRE(splts[6].load_value == 1);
    REQUIRE(splts[7].load_value == 1);

    Patch p = Patch::merge_patch(splts);

    REQUIRE_NAMED("patch are equal", p == check_patch);
}

TestStart(Unittest, "shamrock/patch/Patch.cpp:SplitCoord", splitcoord, 1) {

    using namespace shamrock::patch;

    Patch p0{};
    p0.id_patch        = 0;
    p0.pack_node_index = u64_max;
    p0.load_value      = 8;
    p0.coord_min[0]    = 0;
    p0.coord_min[1]    = 0;
    p0.coord_min[2]    = 0;
    p0.coord_max[0]    = 256;
    p0.coord_max[1]    = 128;
    p0.coord_max[2]    = 1024;
    p0.node_owner_id   = 0;

    u64 min_x = p0.coord_min[0];
    u64 min_y = p0.coord_min[1];
    u64 min_z = p0.coord_min[2];

    u64 split_x = (((p0.coord_max[0] - p0.coord_min[0]) + 1) / 2) - 1 + min_x;
    u64 split_y = (((p0.coord_max[1] - p0.coord_min[1]) + 1) / 2) - 1 + min_y;
    u64 split_z = (((p0.coord_max[2] - p0.coord_min[2]) + 1) / 2) - 1 + min_z;

    u64 max_x = p0.coord_max[0];
    u64 max_y = p0.coord_max[1];
    u64 max_z = p0.coord_max[2];

    std::array<u64, 3> split_out = p0.get_split_coord();

    REQUIRE_NAMED("split 0", split_x == split_out[0]);
    REQUIRE_NAMED("split 1", split_y == split_out[1]);
    REQUIRE_NAMED("split 2", split_z == split_out[2]);
}
