// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file io.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/collective/indexing.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/mpiErrorCheck.hpp"

namespace shamalgs::collective {

    /**
     * @brief Writes data to an MPI file in a collective manner.
     *
     * @tparam T The type of data to be written.
     * @param fh The MPI file handle.
     * @param ptr_data Pointer to the data to be written.
     * @param data_cnt Number of elements of type T to be written.
     * @param file_head_ptr The current file head pointer, which is updated after the write
     * operation.
     *
     */
    template<class T>
    void viewed_write_all_fetch(MPI_File fh, T *ptr_data, u64 data_cnt, u64 &file_head_ptr) {
        auto dtype = get_mpi_type<T>();

        i32 sz;
        MPICHECK(MPI_Type_size(dtype, &sz));

        ViewInfo view = fetch_view(u64(sz) * data_cnt);

        u64 disp = file_head_ptr + view.head_offset;

        MPICHECK(MPI_File_set_view(fh, disp, dtype, dtype, "native", MPI_INFO_NULL));

        MPICHECK(MPI_File_write_all(fh, ptr_data, data_cnt, dtype, MPI_STATUS_IGNORE));

        file_head_ptr = view.total_byte_count + file_head_ptr;
    }

    /**
     * @brief Writes data to an MPI file in a collective manner and updates the file head pointer.
     *
     * @tparam T The type of data to be written.
     * @param fh The MPI file handle.
     * @param ptr_data Pointer to the data to be written.
     * @param data_cnt Number of elements of type T to be written.
     * @param total_cnt Total number of elements of type T in the file.
     * @param file_head_ptr The current file head pointer, which is updated after the write
     * operation.
     *
     */
    template<class T>
    void viewed_write_all_fetch_known_total_size(
        MPI_File fh, T *ptr_data, u64 data_cnt, u64 total_cnt, u64 &file_head_ptr) {
        auto dtype = get_mpi_type<T>();

        i32 sz;
        MPICHECK(MPI_Type_size(dtype, &sz));

        ViewInfo view = fetch_view_known_total(u64(sz) * data_cnt, u64(sz) * total_cnt);

        u64 disp = file_head_ptr + view.head_offset;

        MPICHECK(MPI_File_set_view(fh, disp, dtype, dtype, "native", MPI_INFO_NULL));

        MPICHECK(MPI_File_write_all(fh, ptr_data, data_cnt, dtype, MPI_STATUS_IGNORE));

        file_head_ptr = view.total_byte_count + file_head_ptr;
    }

    /**
     * @brief Writes a string to a file using MPI and updates the file head pointer.
     *
     * Note that all processes should call this function with the same content
     *
     * @param fh MPI file handle
     * @param s string to write
     * @param file_head_ptr file head pointer
     */
    inline void write_header_raw(MPI_File fh, std::string s, u64 &file_head_ptr) {

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        if (shamcomm::world_rank() == 0) {
            MPICHECK(MPI_File_write(fh, s.c_str(), s.size(), MPI_CHAR, MPI_STATUS_IGNORE));
        }

        file_head_ptr = file_head_ptr + s.size();
    }

    /**
     * @brief Reads a string of length len from a file using MPI and updates the file head pointer.
     *
     * @param fh MPI file handle
     * @param len length of string to read
     * @param file_head_ptr file head pointer
     * @return string read from file
     */
    inline std::string read_header_raw(MPI_File fh, size_t len, u64 &file_head_ptr) {

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));
        std::string s;
        s.resize(len);

        MPICHECK(MPI_File_read(fh, s.data(), s.size(), MPI_CHAR, MPI_STATUS_IGNORE));

        file_head_ptr = file_head_ptr + s.size();

        return s;
    }

    /**
     * @brief Writes a size_t to a file using MPI and updates the file head pointer.
     *
     * @param fh MPI file handle
     * @param val value to write
     * @param file_head_ptr file head pointer
     */
    inline void write_header_val(MPI_File fh, size_t val, u64 &file_head_ptr) {

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        if (shamcomm::world_rank() == 0) {
            MPICHECK(MPI_File_write(fh, &val, 1, get_mpi_type<size_t>(), MPI_STATUS_IGNORE));
        }

        file_head_ptr = file_head_ptr + sizeof(size_t);
    }

    /**
     * @brief Writes data at a given offset in a file using MPI.
     *
     * @param fh MPI file handle
     * @param buf pointer to the data to be written
     * @param len number of elements of type T to be written
     * @param file_head_ptr offset in the file where the data should be written
     */
    template<class T>
    inline void write_at(MPI_File fh, const void *buf, size_t len, u64 file_head_ptr) {

        MPICHECK(
            MPI_File_write_at(fh, file_head_ptr, buf, len, get_mpi_type<T>(), MPI_STATUS_IGNORE));
    }

    /**
     * @brief Reads data at a given offset in a file using MPI.
     *
     * @param fh MPI file handle
     * @param buf pointer to the data that should be read
     * @param len number of elements of type T to be read
     * @param file_head_ptr offset in the file where the data should be read
     */
    template<class T>
    inline void read_at(MPI_File fh, void *buf, size_t len, u64 file_head_ptr) {

        MPICHECK(
            MPI_File_read_at(fh, file_head_ptr, buf, len, get_mpi_type<T>(), MPI_STATUS_IGNORE));
    }

    /**
     * @brief Reads a size_t from a file using MPI and updates the file head pointer.
     *
     * @param fh MPI file handle
     * @param file_head_ptr file head pointer
     * @return size_t read from file
     */
    inline size_t read_header_val(MPI_File fh, u64 &file_head_ptr) {

        size_t val = 0;
        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        MPICHECK(MPI_File_read(fh, &val, 1, get_mpi_type<size_t>(), MPI_STATUS_IGNORE));

        file_head_ptr = file_head_ptr + sizeof(size_t);

        return val;
    }

    /**
     * @brief Writes a string to a file using MPI and updates the file head pointer.
     *        The string is preceded by its length.
     *
     * @param fh MPI file handle
     * @param s string to write
     * @param file_head_ptr file head pointer
     */
    inline void write_header(MPI_File fh, std::string s, u64 &file_head_ptr) {

        write_header_val(fh, s.size(), file_head_ptr);
        write_header_raw(fh, s, file_head_ptr);
    }

    /**
     * @brief Reads a string from a file using MPI and updates the file head pointer.
     *        The string is preceded by its length.
     *
     * @param fh MPI file handle
     * @param file_head_ptr file head pointer
     * @return string read from file
     */
    inline std::string read_header(MPI_File fh, u64 &file_head_ptr) {

        size_t len    = read_header_val(fh, file_head_ptr);
        std::string s = read_header_raw(fh, len, file_head_ptr);
        return s;
    }

} // namespace shamalgs::collective
