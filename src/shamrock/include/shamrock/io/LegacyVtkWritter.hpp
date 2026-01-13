// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file LegacyVtkWritter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/endian.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/io.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/memory/bufferFlattening.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/io.hpp"
#include "shamrock/io/details/bufToVtkBuf.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <fstream>
#include <sstream>
#include <string>

namespace shamrock {
    namespace details {

        template<class T>
        using repr_t = typename shambase::VectorProperties<T>::component_type;

        template<class T>
        static constexpr u32 repr_count = shambase::VectorProperties<T>::dimension;

        template<class RT, class T>
        inline void write_buffer_vtktype(
            MPI_File fh,
            sycl::buffer<T> &buf,
            u32 len,
            u32 sum_len,
            bool device_alloc,
            u64 &file_head_ptr) {
            StackEntry stack_loc{};

            if (len == 0) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Cannot call this function with null buffer length");
            }

            const u32 new_cnt     = len * repr_count<T>;
            const u32 new_cnt_sum = sum_len * repr_count<T>;

            shamlog_debug_mpi_ln("VTK write", new_cnt, new_cnt_sum);

            sycl::queue &q = shamsys::instance::get_compute_queue();

            sycl::buffer<RT> buf_w = shamrock::details::to_vtk_buf_type<RT>(q, buf, len);

            RT *usm_buf;
            if (device_alloc) {

                usm_buf = sycl::malloc_device<RT>(new_cnt, q);

                auto ev = q.submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_buf{buf_w, cgh, sycl::read_only};
                    RT *ptr = usm_buf;
                    cgh.parallel_for(sycl::range<1>{new_cnt}, [=](sycl::item<1> i) {
                        ptr[i] = (acc_buf[i]);
                    });
                });
                ev.wait(); // TODO wait for the event only when doing MPI calls

            } else {
                usm_buf = sycl::malloc_host<RT>(new_cnt, q);

                {
                    sycl::host_accessor acc_buf{buf_w, sycl::read_only};
                    for (u32 i = 0; i < new_cnt; i++) {
                        usm_buf[i] = (acc_buf[i]);
                    }
                }
            }

            shamlog_debug_mpi_ln("VTK write", new_cnt);

            shamalgs::collective::viewed_write_all_fetch_known_total_size(
                fh, usm_buf, new_cnt, new_cnt_sum, file_head_ptr);

            sycl::free(usm_buf, q);
        }

        template<class RT, class T>
        inline void write_buffer_vtktype_no_buf(
            MPI_File fh, u32 sum_len, bool device_alloc, u64 &file_head_ptr) {
            StackEntry stack_loc{};

            const u32 new_cnt_sum = sum_len * repr_count<T>;

            shamlog_debug_mpi_ln("VTK write", new_cnt_sum);

            sycl::queue &q = shamsys::instance::get_compute_queue();

            shamalgs::collective::viewed_write_all_fetch_known_total_size<RT>(
                fh, nullptr, 0, new_cnt_sum, file_head_ptr);
        }
    } // namespace details

    enum DataSetTypes { UnstructuredGrid };

    class LegacyVtkWritter {
        MPI_File mfile{};
        std::string fname;
        bool binary;

        u64 file_head_ptr;

        shambase::Timer timer;

        private:
        inline void head_write(std::string s) {
            shamalgs::collective::write_header_raw(mfile, s, file_head_ptr);
        }

        template<class T>
        inline void write_buf(sycl::buffer<T> &buf, u32 len, u32 sum_len) {
            if constexpr (shambase::VectorProperties<T>::is_float_based) {
                details::write_buffer_vtktype<f32>(mfile, buf, len, sum_len, false, file_head_ptr);
            } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
                details::write_buffer_vtktype<i32>(mfile, buf, len, sum_len, false, file_head_ptr);
            } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
                details::write_buffer_vtktype<i32>(mfile, buf, len, sum_len, false, file_head_ptr);
            }
        }

        template<class T>
        inline void write_buf_no_buf(u32 sum_len) {
            if constexpr (shambase::VectorProperties<T>::is_float_based) {
                details::write_buffer_vtktype_no_buf<f32, T>(mfile, sum_len, false, file_head_ptr);
            } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
                details::write_buffer_vtktype_no_buf<i32, T>(mfile, sum_len, false, file_head_ptr);
            } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
                details::write_buffer_vtktype_no_buf<i32, T>(mfile, sum_len, false, file_head_ptr);
            }
        }

        template<class T>
        inline std::string get_buf_type_name() {
            if constexpr (shambase::VectorProperties<T>::is_float_based) {
                return "float";
            } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
                return "int";
            } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
                return "int";
            } else {
                return "unknown";
            }
        }

        u64 points_count;
        bool has_written_points = false;

        u64 cells_count;
        bool has_written_cells = false;

        public:
        inline LegacyVtkWritter(std::string fname, bool binary, DataSetTypes type)
            : fname(fname), binary(binary), file_head_ptr(0_u64) {

            StackEntry stack_loc{};

            timer.start();

            shamlog_debug_ln("VtkWritter", "opening :", fname);

            if (fname.find(".vtk") == std::string::npos) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "the extension should be .vtk");
            }

            shamcomm::open_reset_file(mfile, fname);

            std::stringstream ss;

            if (binary) {
                ss << ("# vtk DataFile Version 4.2\nvtk output\nBINARY\n");
            } else {
                ss << ("# vtk DataFile Version 4.2\nvtk output\nASCII\n");
            }

            if (type == UnstructuredGrid) {
                ss << ("DATASET UNSTRUCTURED_GRID");
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>("unknown dataset type");
            }

            std::string write_str = ss.str();

            head_write(write_str);
        }

        template<class T>
        void write_points(sycl::buffer<sycl::vec<T, 3>> &buf, u32 len) {
            StackEntry stack_loc{};

            shamlog_debug_mpi_ln("VTK write", "write_points");

            u32 sum_len = shamalgs::collective::allreduce_sum(len);

            std::stringstream ss;
            ss << "\n\nPOINTS ";
            ss << sum_len;
            ss << " " << get_buf_type_name<sycl::vec<T, 3>>();
            ss << "\n";

            head_write(ss.str());

            write_buf(buf, len, sum_len);

            has_written_points = true;
            points_count       = sum_len;
        }

        template<class T>
        void write_points_no_buf() {
            StackEntry stack_loc{};

            shamlog_debug_mpi_ln("VTK write", "write_points no buf");

            u32 sum_len = shamalgs::collective::allreduce_sum(0);

            std::stringstream ss;
            ss << "\n\nPOINTS ";
            ss << sum_len;
            ss << " " << get_buf_type_name<sycl::vec<T, 3>>();
            ss << "\n";

            head_write(ss.str());

            write_buf_no_buf<T>(sum_len);

            has_written_points = true;
            points_count       = sum_len;
        }

        template<class T>
        void write_points(std::unique_ptr<sycl::buffer<sycl::vec<T, 3>>> &buf, u32 len) {
            if (len > 0) {
                write_points(shambase::get_check_ref(buf), len);
            } else {
                write_points_no_buf<T>();
            }
        }

        template<class T>
        void write_voxel_cells(
            sycl::buffer<sycl::vec<T, 3>> &buf_min,
            sycl::buffer<sycl::vec<T, 3>> &buf_max,
            u32 len) {

            sycl::buffer<sycl::vec<T, 3>> pos_points(len * 8);

            auto view      = shamalgs::collective::fetch_view(len);
            u32 sum_len    = view.total_byte_count;
            u32 len_offset = view.head_offset;

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_min{buf_min, cgh, sycl::read_only};
                sycl::accessor acc_max{buf_max, cgh, sycl::read_only};

                sycl::accessor acc_points{pos_points, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                    u32 idx = id.get_linear_id() * 8;

                    sycl::vec<T, 3> pmin = acc_min[id];
                    sycl::vec<T, 3> pmax = acc_max[id];

                    acc_points[idx + 0] = pmin;
                    acc_points[idx + 1] = {pmax.x(), pmin.y(), pmin.z()};
                    acc_points[idx + 2] = {pmin.x(), pmax.y(), pmin.z()};
                    acc_points[idx + 3] = {pmax.x(), pmax.y(), pmin.z()};
                    acc_points[idx + 4] = {pmin.x(), pmin.y(), pmax.z()};
                    acc_points[idx + 5] = {pmax.x(), pmin.y(), pmax.z()};
                    acc_points[idx + 6] = {pmin.x(), pmax.y(), pmax.z()};
                    acc_points[idx + 7] = pmax;
                });
            });

            write_points(pos_points, len * 8);

            std::stringstream ss;
            ss << "\n\nCELLS ";
            ss << sum_len;
            ss << " " << sum_len * 9;
            ss << "\n";
            head_write(ss.str());

            sycl::buffer<i32> idx_cells(len * 9);
            sycl::buffer<i32> type_cell(len);

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor idxs{idx_cells, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor cellt{type_cell, cgh, sycl::write_only, sycl::no_init};

                u32 idp_off = len_offset * 8;

                cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> item) {
                    u32 idp = item.get_linear_id() * 8;
                    u32 idx = item.get_linear_id() * 9;

                    idxs[idx + 0] = 8;
                    idxs[idx + 1] = idp_off + idp + 0;
                    idxs[idx + 2] = idp_off + idp + 1;
                    idxs[idx + 3] = idp_off + idp + 2;
                    idxs[idx + 4] = idp_off + idp + 3;
                    idxs[idx + 5] = idp_off + idp + 4;
                    idxs[idx + 6] = idp_off + idp + 5;
                    idxs[idx + 7] = idp_off + idp + 6;
                    idxs[idx + 8] = idp_off + idp + 7;

                    cellt[item] = 11;
                });
            });

            write_buf(idx_cells, len * 9, sum_len * 9);

            std::stringstream ss2;
            ss2 << "\n\nCELL_TYPES ";
            ss2 << sum_len;
            ss2 << "\n";
            head_write(ss2.str());

            write_buf(type_cell, len, sum_len);

            cells_count       = sum_len;
            has_written_cells = true;
        }

        void add_point_data_section() {

            if (!has_written_points) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "no points had been written");
            }

            std::stringstream ss;
            ss << "\n\nPOINT_DATA ";
            ss << points_count;

            head_write(ss.str());
        }

        void add_cell_data_section() {

            if (!has_written_cells) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "no cells had been written");
            }

            std::stringstream ss;
            ss << "\n\nCELL_DATA ";
            ss << cells_count;

            head_write(ss.str());
        }

        void add_field_data_section(u32 num_field) {

            if (!has_written_points) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "no points had been written");
            }

            std::stringstream ss;
            ss << "\nFIELD FieldData ";
            ss << num_field;

            head_write(ss.str());
        }

        template<class T>
        void write_field(std::string name, sycl::buffer<T> &buf, u32 len) {

            u32 sum_len = shamalgs::collective::allreduce_sum(len);

            std::stringstream ss;
            ss << "\n" << name;
            ss << " " << details::repr_count<T>;
            ss << " " << sum_len;
            ss << " " << get_buf_type_name<T>();
            ss << "\n";
            head_write(ss.str());

            write_buf(buf, len, sum_len);
        }

        template<class T>
        void write_field_no_buf(std::string name) {

            u32 sum_len = shamalgs::collective::allreduce_sum(0);

            std::stringstream ss;
            ss << "\n" << name;
            ss << " " << details::repr_count<T>;
            ss << " " << sum_len;
            ss << " " << get_buf_type_name<T>();
            ss << "\n";
            head_write(ss.str());

            write_buf_no_buf<T>(sum_len);
        }

        template<class T>
        void write_field(std::string name, std::unique_ptr<sycl::buffer<T>> &buf, u32 len) {
            if (len > 0) {
                sycl::buffer<T> &buf_ref = shambase::get_check_ref(buf);
                if (buf_ref.size() < len) {
                    shambase::throw_with_loc<std::runtime_error>(shambase::format(
                        "the buffer is smaller than expected write field size\n    buf size = {}, "
                        "cnt = {}",
                        buf_ref.size(),
                        len));
                }
                write_field(name, buf_ref, len);
            } else {
                write_field_no_buf<T>(name);
            }
        }

        inline ~LegacyVtkWritter() {
            shamlog_debug_mpi_ln("LegacyVtkWritter", "calling : shamcomm::mpi::File_close");
            shamcomm::mpi::File_close(&mfile);
            timer.end();

            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "VTK Dump",
                    shambase::format(
                        "dump to {}\n              - took {}, bandwidth = {}/s",
                        fname,
                        timer.get_time_str(),
                        shambase::readable_sizeof(file_head_ptr / timer.elasped_sec())));
            }
        }

        LegacyVtkWritter(const LegacyVtkWritter &)            = delete;
        LegacyVtkWritter &operator=(const LegacyVtkWritter &) = delete;
        LegacyVtkWritter(LegacyVtkWritter &&other)
            : mfile(other.mfile), fname(std::move(other.fname)), binary(other.binary),
              file_head_ptr(other.file_head_ptr) {}                     // move constructor
        LegacyVtkWritter &operator=(LegacyVtkWritter &&other) = delete; // move assignment
    };
} // namespace shamrock
