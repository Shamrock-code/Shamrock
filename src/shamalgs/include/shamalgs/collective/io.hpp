// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/SyclMpiTypes.hpp"
#include "shamcomm/mpiErrorCheck.hpp"

namespace shamalgs::collective {

    /**
     * @brief
     *
     * @tparam T
     * @param ptr_data
     * @param data_cnt
     * @param file_head_ptr
     * @return u64 the new file head ptr
     */
    template<class T>
    void viewed_write_all_fetch(MPI_File fh, T *ptr_data, u64 data_cnt, u64 & file_head_ptr) {
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
     * @brief
     *
     * @tparam T
     * @param ptr_data
     * @param data_cnt
     * @param file_head_ptr
     * @return u64 the new file head ptr
     */
    template<class T>
    void viewed_write_all_fetch_known_total_size(MPI_File fh, T *ptr_data, u64 data_cnt, u64 total_cnt, u64 & file_head_ptr) {
        auto dtype = get_mpi_type<T>();

        i32 sz;
        MPICHECK(MPI_Type_size(dtype, &sz));

        ViewInfo view = fetch_view_known_total(u64(sz) * data_cnt, u64(sz)*total_cnt);

        u64 disp = file_head_ptr + view.head_offset;

        MPICHECK(MPI_File_set_view(fh, disp, dtype, dtype, "native", MPI_INFO_NULL));

        MPICHECK(MPI_File_write_all(fh, ptr_data, data_cnt, dtype, MPI_STATUS_IGNORE));

        file_head_ptr = view.total_byte_count + file_head_ptr;
    }



    inline void write_header_raw(MPI_File fh, std::string s, u64 & file_head_ptr) {

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        if (shamcomm::world_rank() == 0) {
            MPICHECK(MPI_File_write(fh, s.c_str(), s.size(), MPI_CHAR, MPI_STATUS_IGNORE));
        }

        file_head_ptr = file_head_ptr + s.size();
    }

    inline std::string read_header_raw(MPI_File fh, size_t len, u64 & file_head_ptr) {

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));
        std::string s;
        s.resize(len);

            MPICHECK(MPI_File_read(fh, s.data(), s.size(), MPI_CHAR, MPI_STATUS_IGNORE));

        file_head_ptr = file_head_ptr + s.size();

        return s;
    }


    inline void write_header_val(MPI_File fh, size_t val, u64 & file_head_ptr){

        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        if (shamcomm::world_rank() == 0) {
            MPICHECK(MPI_File_write(fh, &val, 1, get_mpi_type<size_t>(), MPI_STATUS_IGNORE));
        }

        file_head_ptr = file_head_ptr + sizeof(size_t);
    }

    inline size_t read_header_val(MPI_File fh, u64 & file_head_ptr){
        
        size_t val = 0;
        MPICHECK(MPI_File_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL));

        MPICHECK(MPI_File_read(fh, &val, 1, get_mpi_type<size_t>(), MPI_STATUS_IGNORE));

        file_head_ptr = file_head_ptr + sizeof(size_t);

        return val;
    }


    inline void write_header(MPI_File fh,std::string s, u64 & file_head_ptr){

        write_header_val(fh, s.size(), file_head_ptr);
        write_header_raw(fh, s, file_head_ptr);
    }

    inline std::string read_header(MPI_File fh, u64 & file_head_ptr){

        size_t len = read_header_val(fh, file_head_ptr);
        std::string s = read_header_raw(fh, len, file_head_ptr);
        return s;

    }

    

} // namespace shamalgs::collective