// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyCommonUtils.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shamalgs/primitives/compute_histogram.hpp"
#include "shamalgs/primitives/upper_bound.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/MemPerfInfos.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <vector>

// Define the operator += for sham::DeviceBuffer, implementation to be done later.
namespace sham {
    template<typename T>
    DeviceBuffer<T> &operator+=(DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
        sham::kernel_call(
            rhs.get_queue(),
            sham::MultiRef{rhs},
            sham::MultiRef{lhs},
            lhs.get_size(),
            [](u32 n, const T *rhs, T *lhs) {
                lhs[n] += rhs[n];
            });
        return lhs;
    }

    template<typename T>
    DeviceBuffer<T> &operator/=(DeviceBuffer<T> &lhs, const DeviceBuffer<T> &rhs) {
        sham::kernel_call(
            rhs.get_queue(),
            sham::MultiRef{rhs},
            sham::MultiRef{lhs},
            lhs.get_size(),
            [](u32 n, const T *rhs, T *lhs) {
                auto r = rhs[n];
                if (r != 0) {
                    lhs[n] /= r;
                } else {
                    lhs[n] = std::numeric_limits<f64>::quiet_NaN();
                }
            });
        return lhs;
    }

} // namespace sham

Register_pymod(shammodelcommonlibinit) {

    m.def(
        "compute_histogram",
        [](std::vector<f64> bin_edges,
           shamrock::solvergraph::Field<f64> &x_field,
           shamrock::solvergraph::Field<f64> &y_field,
           bool do_average) -> std::vector<f64> {
            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            u32 nx = bin_edges.size() - 1;
            std::vector<f64> bin_edge_inf(nx);
            std::vector<f64> bin_edge_sup(nx);

            for (size_t i = 0; i < nx; i++) {
                bin_edge_inf[i] = bin_edges[i];
                bin_edge_sup[i] = bin_edges[i + 1];
            }

            sham::DeviceBuffer<f64> ret(bin_edge_inf.size(), dev_sched);
            ret.fill(0);

            sham::DeviceBuffer<f64> bin_inf(bin_edge_inf.size(), dev_sched);
            sham::DeviceBuffer<f64> bin_sup(bin_edge_inf.size(), dev_sched);
            bin_inf.copy_from_stdvec(bin_edge_inf);
            bin_sup.copy_from_stdvec(bin_edge_sup);

            shambase::DistributedData<u32> obj_cnts = x_field.get_obj_cnts();

            obj_cnts.for_each([&](u64 id_patch, const unsigned int &obj_cnt) {
                ret += shamalgs::primitives::compute_histogram<f64>(
                    dev_sched,
                    bin_inf,
                    bin_sup,
                    obj_cnt,
                    [](const f64 &bin_edge_inf,
                       const f64 &bin_edge_sup,
                       const f64 &x_val,
                       const f64 &y_val,
                       bool &has_value) {
                        has_value = x_val >= bin_edge_inf && x_val < bin_edge_sup;
                        return has_value ? y_val : 0;
                    },
                    x_field.get_buf(id_patch),
                    y_field.get_buf(id_patch));
            });

            shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

            if (do_average) {

                sham::DeviceBuffer<f64> norm(bin_edge_inf.size(), dev_sched);
                norm.fill(0);

                obj_cnts.for_each([&](u64 id_patch, const unsigned int &obj_cnt) {
                    sham::DeviceBuffer<f64> unit_buf(obj_cnt, dev_sched);
                    unit_buf.fill(1);

                    norm += shamalgs::primitives::compute_histogram<f64>(
                        dev_sched,
                        bin_inf,
                        bin_sup,
                        obj_cnt,
                        [](const f64 &bin_edge_inf,
                           const f64 &bin_edge_sup,
                           const f64 &x_val,
                           const f64 &y_val,
                           bool &has_value) {
                            has_value = x_val >= bin_edge_inf && x_val < bin_edge_sup;
                            return has_value ? y_val : 0;
                        },
                        x_field.get_buf(id_patch),
                        unit_buf);
                });

                shamalgs::collective::reduce_buffer_in_place_sum(norm, MPI_COMM_WORLD);

                ret /= norm;
            }

            return ret.copy_to_stdvec();
        },
        py::kw_only{},
        py::arg("bin_edges"),
        py::arg("x_field"),
        py::arg("y_field"),
        py::arg("do_average") = false);

    m.def(
        "compute_histogram_convolve_x",
        [](std::vector<f64> bin_edges,
           shamrock::solvergraph::Field<f64> &x_field,
           shamrock::solvergraph::Field<f64> &y_field,
           shamrock::solvergraph::Field<f64> &size_field,
           bool do_average) -> std::vector<f64> {
            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            u32 nx = bin_edges.size() - 1;
            std::vector<f64> bin_edge_inf(nx);
            std::vector<f64> bin_edge_sup(nx);

            for (size_t i = 0; i < nx; i++) {
                bin_edge_inf[i] = bin_edges[i];
                bin_edge_sup[i] = bin_edges[i + 1];
            }

            sham::DeviceBuffer<f64> ret(bin_edge_inf.size(), dev_sched);
            ret.fill(0);

            sham::DeviceBuffer<f64> bin_inf(bin_edge_inf.size(), dev_sched);
            sham::DeviceBuffer<f64> bin_sup(bin_edge_inf.size(), dev_sched);
            bin_inf.copy_from_stdvec(bin_edge_inf);
            bin_sup.copy_from_stdvec(bin_edge_sup);

            shambase::DistributedData<u32> obj_cnts = x_field.get_obj_cnts();

            obj_cnts.for_each([&](u64 id_patch, const unsigned int &obj_cnt) {
                ret += shamalgs::primitives::compute_histogram<f64>(
                    dev_sched,
                    bin_inf,
                    bin_sup,
                    obj_cnt,
                    [](const f64 &bin_edge_inf,
                       const f64 &bin_edge_sup,
                       const f64 &x_val,
                       const f64 &y_val,
                       const f64 &size_val,
                       bool &has_value) {
                        has_value
                            = x_val >= bin_edge_inf - size_val && x_val < bin_edge_sup + size_val;
                        return has_value ? y_val : 0;
                    },
                    x_field.get_buf(id_patch),
                    y_field.get_buf(id_patch),
                    size_field.get_buf(id_patch));
            });

            shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

            if (do_average) {

                sham::DeviceBuffer<f64> norm(bin_edge_inf.size(), dev_sched);
                norm.fill(0);

                obj_cnts.for_each([&](u64 id_patch, const unsigned int &obj_cnt) {
                    sham::DeviceBuffer<f64> unit_buf(obj_cnt, dev_sched);
                    unit_buf.fill(1);

                    norm += shamalgs::primitives::compute_histogram<f64>(
                        dev_sched,
                        bin_inf,
                        bin_sup,
                        obj_cnt,
                        [](const f64 &bin_edge_inf,
                           const f64 &bin_edge_sup,
                           const f64 &x_val,
                           const f64 &y_val,
                           const f64 &size_val,
                           bool &has_value) {
                            has_value = x_val >= bin_edge_inf - size_val
                                        && x_val < bin_edge_sup + size_val;
                            return has_value ? y_val : 0;
                        },
                        x_field.get_buf(id_patch),
                        unit_buf,
                        size_field.get_buf(id_patch));
                });

                shamalgs::collective::reduce_buffer_in_place_sum(norm, MPI_COMM_WORLD);

                ret /= norm;
            }

            return ret.copy_to_stdvec();
        },
        py::kw_only{},
        py::arg("bin_edges"),
        py::arg("x_field"),
        py::arg("y_field"),
        py::arg("size_field"),
        py::arg("do_average") = false);

    m.def(
        "compute_histogram_2d",
        [](std::vector<f64> bin_edges_x,
           std::vector<f64> bin_edges_y,
           shamrock::solvergraph::Field<f64> &x_field,
           shamrock::solvergraph::Field<f64> &y_field) -> std::vector<u64> {
            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            u32 nx = bin_edges_x.size() - 1;
            u32 ny = bin_edges_y.size() - 1;

            sham::DeviceBuffer<u64> ret(nx * ny, dev_sched);
            ret.fill(0);

            shambase::DistributedData<u32> obj_cnts = x_field.get_obj_cnts();

            sham::DeviceBuffer<f64> binsx(bin_edges_x.size(), dev_sched);
            sham::DeviceBuffer<f64> binsy(bin_edges_y.size(), dev_sched);
            binsx.copy_from_stdvec(bin_edges_x);
            binsy.copy_from_stdvec(bin_edges_y);

            obj_cnts.for_each([&](u64 id_patch, const unsigned int &obj_cnt) {
                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{
                        binsx, binsy, x_field.get_buf(id_patch), y_field.get_buf(id_patch)},
                    sham::MultiRef{ret},
                    obj_cnt,
                    [nx, ny](
                        u32 id,
                        const f64 *__restrict x_bins,
                        const f64 *__restrict y_bins,
                        const f64 *__restrict x_field,
                        const f64 *__restrict y_field,
                        u64 *__restrict pic) {
                        auto get_pic_coord = [&](u32 ix, u32 iy) {
                            return ix + iy * nx;
                        };

                        u32 ix = shamalgs::primitives::binary_search_upper_bound(
                            x_bins, 0, nx + 1, x_field[id]);
                        u32 iy = shamalgs::primitives::binary_search_upper_bound(
                            y_bins, 0, ny + 1, y_field[id]);

                        using atomic_ref_T = sycl::atomic_ref<
                            u64,
                            sycl::memory_order_relaxed,
                            sycl::memory_scope_device,
                            sycl::access::address_space::global_space>;

                        atomic_ref_T pic_ref(pic[get_pic_coord(ix, iy)]);
                        pic_ref++;
                    });
            });

            shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

            return ret.copy_to_stdvec();
        },
        py::kw_only{},
        py::arg("bin_edges_x"),
        py::arg("bin_edges_y"),
        py::arg("x_field"),
        py::arg("y_field"));
}
