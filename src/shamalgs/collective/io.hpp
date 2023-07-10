// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/indexing.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"

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
        mpi::type_size(dtype, &sz);

        ViewInfo view = fetch_view(u64(sz) * data_cnt);

        u64 disp = file_head_ptr + view.head_offset;

        mpi::file_set_view(fh, disp, dtype, dtype, "native", MPI_INFO_NULL);

        mpi::file_write_all(fh, ptr_data, data_cnt, dtype, MPI_STATUS_IGNORE);

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
        mpi::type_size(dtype, &sz);

        ViewInfo view = fetch_view_known_total(u64(sz) * data_cnt, u64(sz)*total_cnt);

        u64 disp = file_head_ptr + view.head_offset;

        mpi::file_set_view(fh, disp, dtype, dtype, "native", MPI_INFO_NULL);

        mpi::file_write_all(fh, ptr_data, data_cnt, dtype, MPI_STATUS_IGNORE);

        file_head_ptr = view.total_byte_count + file_head_ptr;
    }



    inline void write_header(MPI_File fh, std::string s, u64 & file_head_ptr) {

        mpi::file_set_view(fh, file_head_ptr, MPI_BYTE, MPI_CHAR, "native", MPI_INFO_NULL);

        if (shamsys::instance::world_rank == 0) {
            mpi::file_write(fh, s.c_str(), s.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        }

        file_head_ptr = file_head_ptr + s.size();
    }

    /**
     * @brief open a mpi file and remove its content
     * 
     */
    inline void open_reset_file(MPI_File & fh, std::string fname){

        int rc = mpi::file_open(
            MPI_COMM_WORLD,
            fname.c_str(),
            MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN,
            MPI_INFO_NULL,
            &fh
        );

        if(rc != MPI_SUCCESS){ 

            if(shamsys::instance::world_rank == 0){
                mpi::file_delete(fname.c_str(), MPI_INFO_NULL);
            }

            int rc = mpi::file_open(
                MPI_COMM_WORLD,
                fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN,
                MPI_INFO_NULL,
                &fh
            );

            if(rc != MPI_SUCCESS){
                throw shambase::throw_with_loc<std::runtime_error>("cannot create file");
            }

        }
    }


    

} // namespace shamalgs::collective