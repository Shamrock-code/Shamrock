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

    constexpr size_t mock_vector_tile_size = 65536;
    static_assert(mock_vector_tile_size <= std::numeric_limits<u32>::max());

    /// Random tile of mock_vector_tile_size bytes, repeated to build/check large buffers
    std::vector<u8> mock_tile(u64 seed) {
        return shamalgs::primitives::mock_vector<u8>(seed, mock_vector_tile_size, u8{0}, u8{255});
    }

    /// Fill buf with the tile generated from seed, repeated
    void fill_tiled(u64 seed, std::vector<u8> &buf) {
        const std::vector<u8> tile = mock_tile(seed);
        for (size_t off = 0; off < buf.size(); off += tile.size()) {
            const size_t n = std::min<size_t>(tile.size(), buf.size() - off);
            std::memcpy(buf.data() + off, tile.data(), n);
        }
    }

    /// Check that buf contains the tile generated from seed, repeated
    bool is_tiled(u64 seed, const std::vector<u8> &buf) {
        const std::vector<u8> tile = mock_tile(seed);
        for (size_t off = 0; off < buf.size(); off += tile.size()) {
            const size_t n = std::min<size_t>(tile.size(), buf.size() - off);
            if (std::memcmp(buf.data() + off, tile.data(), n) != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Write two consecutive chunks to a file, read them back and check them
     *
     * To keep the memory footprint low (this is used with chunks larger than 2 GiB), the chunk
     * contents are a repeated random tile: a single buffer is allocated for the largest chunk and
     * is reused for writing then reading both chunks, the read data being checked against the
     * regenerated tile instead of a stored copy. The file is deleted at the end.
     */
    template<class WriteFn, class ReadFn>
    void test_read_write_at_chunks(
        const std::string &fname,
        size_t chunk0_len,
        size_t chunk1_len,
        WriteFn &&write_fn,
        ReadFn &&read_fn) {

        constexpr u64 seed0 = 0x1111;
        constexpr u64 seed1 = 0x2222;

        std::vector<u8> buf;
        if (shamcomm::world_rank() == 0) {
            try {
                buf.reserve(std::max(chunk0_len, chunk1_len));
            } catch (const std::bad_alloc &) {
                REQUIRE(false);
                return;
            }
        }

        // resize (within the reserved capacity) and zero the buffer, so that a read that does not
        // fill it cannot match the expected tile
        auto reset_buf = [&](size_t len) {
            buf.assign(len, u8{0});
        };

        { // Write
            MPI_File mfile{};

            shamcomm::open_reset_file(mfile, fname);
            if (shamcomm::world_rank() == 0) {
                reset_buf(chunk0_len);
                fill_tiled(seed0, buf);
                write_fn(mfile, buf.data(), chunk0_len, 0);

                reset_buf(chunk1_len);
                fill_tiled(seed1, buf);
                write_fn(mfile, buf.data(), chunk1_len, chunk0_len);
            }

            MPI_File_close(&mfile);
        }

        { // Read + verify
            MPI_File mfile{};

            shamcomm::open_read_only_file(mfile, fname);
            if (shamcomm::world_rank() == 0) {
                reset_buf(chunk0_len);
                read_fn(mfile, buf.data(), chunk0_len, 0);
                REQUIRE_NAMED("chunk 0 read back matches", is_tiled(seed0, buf));

                reset_buf(chunk1_len);
                read_fn(mfile, buf.data(), chunk1_len, chunk0_len);
                REQUIRE_NAMED("chunk 1 read back matches", is_tiled(seed1, buf));
            }

            MPI_File_close(&mfile);
        }

        if (shamcomm::world_rank() == 0) {
            MPI_File_delete(fname.c_str(), MPI_INFO_NULL);
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
        // chunk 0 must exceed INT_MAX (2^31 - 1) bytes to exercise the large path, and chunk 1 is
        // then written at an offset above 2^31. Both are kept just large enough to limit the
        // memory (one buffer of chunk 0 size) and disk usage on small CI runners.
        2'200'000'000ull,
        12'345'678ull,
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::write_at_large(fh, buf, len, off);
        },
        [](MPI_File fh, u8 *buf, size_t len, u64 off) {
            shamalgs::collective::read_at_large(fh, buf, len, off);
        });
}
