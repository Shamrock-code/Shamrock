// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "vtk-write-parts", vtk_write_test_particles, -1) {

    shamrock::LegacyVtkWritter writter("out-parts.vtk", true, shamrock::UnstructuredGrid);

    f32 offset = shamcomm::world_rank() * shamcomm::world_size();

    std::vector<f32_3> pos = {
        f32_3{0 + offset, 0 + offset, 0 + offset},
        f32_3{1 + offset, 0 + offset, 0 + offset},
        f32_3{0 + offset, 1 + offset, 0 + offset},
        f32_3{0 + offset, 0 + offset, 1 + offset},
        f32_3{1 + offset, 1 + offset, 0 + offset},
    };

    sycl::buffer<f32_3> buf(pos.data(), pos.size());

    writter.write_points(buf, 5);

    writter.add_point_data_section();
    writter.add_field_data_section(2);

    std::vector<f32> field_1 = {0, 1, 2, 3, 4};
    std::vector<f32> field_2 = {11, 9, 7, 15, 20};

    sycl::buffer<f32> buffield_1(field_1.data(), field_1.size());
    sycl::buffer<f32> buffield_2(field_2.data(), field_2.size());

    writter.write_field("field1", buffield_1, 5);
    writter.write_field("field2", buffield_2, 5);
}

TestStart(Unittest, "vtk-write-cells", vtk_write_test_cells, -1) {

    shamrock::LegacyVtkWritter writter("out-cell.vtk", true, shamrock::UnstructuredGrid);

    f32 offset = shamcomm::world_rank() * shamcomm::world_size();

    std::vector<f32_3> pos = {
        f32_3{0 + offset, 0 + offset, 0 + offset},
        f32_3{1 + offset, 0 + offset, 0 + offset},
        f32_3{0 + offset, 1 + offset, 0 + offset},
        f32_3{0 + offset, 0 + offset, 1 + offset},
        f32_3{1 + offset, 1 + offset, 0 + offset},
    };

    std::vector<f32_3> pos2 = {
        f32_3{1 + offset, 1 + offset, 1 + offset},
        f32_3{2 + offset, 1 + offset, 1 + offset},
        f32_3{1 + offset, 2 + offset, 1 + offset},
        f32_3{1 + offset, 1 + offset, 2 + offset},
        f32_3{2 + offset, 2 + offset, 1 + offset},
    };

    sycl::buffer<f32_3> buf(pos.data(), pos.size());
    sycl::buffer<f32_3> buf2(pos2.data(), pos2.size());

    writter.write_voxel_cells(buf, buf2, 5);

    writter.add_cell_data_section();
    writter.add_field_data_section(2);

    std::vector<f32> field_1 = {0, 1, 2, 3, 4};
    std::vector<f32> field_2 = {11, 9, 7, 15, 20};

    sycl::buffer<f32> buffield_1(field_1.data(), field_1.size());
    sycl::buffer<f32> buffield_2(field_2.data(), field_2.size());

    writter.write_field("field1", buffield_1, 5);
    writter.write_field("field2", buffield_2, 5);
}
