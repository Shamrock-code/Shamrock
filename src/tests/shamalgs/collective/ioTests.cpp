// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/io.hpp"
#include "shamcomm/io.hpp"
#include "shamtest/shamtest.hpp"

#define TEST_VAL1 0x12345
#define TEST_VAL2 0x123456
#define TEST_FILE_NAME "iotest"

TestStart(Unittest, "shamalgs/collective/io/header_val", test_collective_io_header_val, -1) {

    { // Write
        u64 head_ptr = 0;

        MPI_File mfile{};

        shamcomm::open_reset_file(mfile, TEST_FILE_NAME);
        shamalgs::collective::write_header_val(mfile, TEST_VAL1, head_ptr);
        shamalgs::collective::write_header_val(mfile, TEST_VAL2, head_ptr);

        MPI_File_close(&mfile);
    }

    { // Read
        u64 head_ptr = 0;

        MPI_File mfile{};

        shamcomm::open_read_only_file(mfile, TEST_FILE_NAME);
        auto read_val1 = shamalgs::collective::read_header_val(mfile, head_ptr);
        auto read_val2 = shamalgs::collective::read_header_val(mfile, head_ptr);
        REQUIRE_EQUAL(read_val1, TEST_VAL1);
        REQUIRE_EQUAL(read_val2, TEST_VAL2);

        MPI_File_close(&mfile);
    }
}

TestStart(Unittest, "shamalgs/collective/io/header_read_write", test_collective_io_header_rw, -1) {

    static std::string ref_str = "dazjndazndzad azd azdijnazidiaz";

    { // Write
        u64 head_ptr = 0;

        MPI_File mfile{};

        shamcomm::open_reset_file(mfile, TEST_FILE_NAME);
        shamalgs::collective::write_header(mfile, ref_str, head_ptr);

        MPI_File_close(&mfile);
    }

    { // Read
        u64 head_ptr = 0;

        MPI_File mfile{};

        shamcomm::open_read_only_file(mfile, TEST_FILE_NAME);
        auto read_str = shamalgs::collective::read_header(mfile, head_ptr);
        REQUIRE(read_str == ref_str);

        MPI_File_close(&mfile);
    }
}
