// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/io.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamcomm/io.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace {

    std::vector<u8> mock_vector_tiled(u64 seed, size_t len) {
        constexpr size_t mock_vector_tile_size = 65536;
        static_assert(mock_vector_tile_size <= std::numeric_limits<u32>::max());

        const size_t tile_len = mock_vector_tile_size;
        std::vector<u8> tile
            = shamalgs::primitives::mock_vector<u8>(seed, tile_len, u8{0}, u8{255});

        std::vector<u8> buf(len);
        for (size_t off = 0; off < len; off += tile_len) {
            const size_t n = std::min<size_t>(tile_len, len - off);
            std::memcpy(buf.data() + off, tile.data(), n);
        }
        return buf;
    }

    template<class WriteFn, class ReadFn>
    void test_read_write_at_chunks(
        const std::string &fname,
        size_t chunk0_len,
        size_t chunk1_len,
        WriteFn &&write_fn,
        ReadFn &&read_fn) {

        std::vector<u8> ref0, ref1;
        if (shamcomm::world_rank() == 0) {
            try {
                ref0 = mock_vector_tiled(0x1111, chunk0_len);
                ref1 = mock_vector_tiled(0x2222, chunk1_len);
            } catch (const std::bad_alloc &) {
                REQUIRE(false);
                return;
            }
        }

        { // Write
            MPI_File mfile{};

            shamcomm::open_reset_file(mfile, fname);
            if (shamcomm::world_rank() == 0) {
                write_fn(mfile, ref0.data(), chunk0_len, 0);
                write_fn(mfile, ref1.data(), chunk1_len, chunk0_len);
            }

            MPI_File_close(&mfile);
        }

        { // Read + verify
            MPI_File mfile{};

            shamcomm::open_read_only_file(mfile, fname);
            if (shamcomm::world_rank() == 0) {
                std::vector<u8> out0(chunk0_len), out1(chunk1_len);
                read_fn(mfile, out0.data(), chunk0_len, 0);
                read_fn(mfile, out1.data(), chunk1_len, chunk0_len);
                REQUIRE_EQUAL(ref0, out0);
                REQUIRE_EQUAL(ref1, out1);
            }

            MPI_File_close(&mfile);
        }
    }

} // namespace

#define TEST_VAL1 0x12345
#define TEST_VAL2 0x123456
#define TEST_FILE_NAME "iotest"

NEW_TEST(Unittest, "shamalgs/collective/io/header_val", -1) {

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

NEW_TEST(Unittest, "shamalgs/collective/io/header_read_write", -1) {

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

NEW_TEST(Unittest, "shamalgs/collective/io/read_write_at", -1) {
    test_read_write_at_chunks(
        "iotest_read_at",
        12345,
        98654,
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::write_at<u8>(fh, buf, len, off);
        },
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::read_at<u8>(fh, buf, len, off);
        });
}

NEW_TEST(Unittest, "shamalgs/collective/io/read_write_at_large", -1) {
    test_read_write_at_chunks(
        "iotest_read_at_large",
        2'600'000'000ull,
        600'000'000ull,
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::write_at_large(fh, buf, len, off);
        },
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::read_at_large(fh, buf, len, off);
        });
}
