// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TreeMortonCodes.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/TreeMortonCodes.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shamtree/RadixTreeMortonBuilder.hpp"

namespace shamrock::tree {

    template<class u_morton>
    template<class T>
    void TreeMortonCodes<u_morton>::build(
        sycl::queue &queue,
        shammath::CoordRange<T> coord_range,
        u32 obj_cnt,
        sycl::buffer<T> &pos_buf) {
        StackEntry stack_loc{};

        this->obj_cnt = obj_cnt;

        using TProp = shambase::VectorProperties<T>;

        RadixTreeMortonBuilder<u_morton, T, TProp::dimension>::build(
            queue,
            {coord_range.lower, coord_range.upper},
            pos_buf,
            obj_cnt,
            buf_morton,
            buf_particle_index_map);
    }

    template<class u_morton>
    template<class T>
    void TreeMortonCodes<u_morton>::build(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<T> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<T> &pos_buf) {
        StackEntry stack_loc{};

        this->obj_cnt = obj_cnt;

        using TProp = shambase::VectorProperties<T>;

        RadixTreeMortonBuilder<u_morton, T, TProp::dimension>::build(
            dev_sched,
            {coord_range.lower, coord_range.upper},
            pos_buf,
            obj_cnt,
            buf_morton,
            buf_particle_index_map);
    }

    template<class u_morton>
    template<class T>
    std::unique_ptr<sycl::buffer<u_morton>> TreeMortonCodes<u_morton>::build_raw(
        sycl::queue &queue,
        shammath::CoordRange<T> coord_range,
        u32 obj_cnt,
        sycl::buffer<T> &pos_buf) {

        std::unique_ptr<sycl::buffer<u_morton>> buf_morton;

        StackEntry stack_loc{};

        using TProp = shambase::VectorProperties<T>;

        RadixTreeMortonBuilder<u_morton, T, TProp::dimension>::build_raw(
            queue, {coord_range.lower, coord_range.upper}, pos_buf, obj_cnt, buf_morton);

        return buf_morton;
    }

    template<class u_morton>
    bool TreeMortonCodes<u_morton>::operator==(const TreeMortonCodes<u_morton> &rhs) const {
        bool cmp = true;

        cmp = cmp && (obj_cnt == rhs.obj_cnt);

        using namespace shamalgs::primitives;

        cmp = cmp
              && equals(
                  shamsys::instance::get_compute_queue(), *buf_morton, *rhs.buf_morton, obj_cnt);
        cmp = cmp
              && equals(
                  shamsys::instance::get_compute_queue(),
                  *buf_particle_index_map,
                  *rhs.buf_particle_index_map,
                  obj_cnt);

        return cmp;
    }

    template class TreeMortonCodes<u32>;
    template class TreeMortonCodes<u64>;

    // u32, f64_3
    template void TreeMortonCodes<u32>::build<f64_3>(
        sycl::queue &queue,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f64_3> &pos_buf);

    template void TreeMortonCodes<u32>::build<f64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<f64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u32>> TreeMortonCodes<u32>::build_raw<f64_3>(
        sycl::queue &queue,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f64_3> &pos_buf);

    // u32, f32_3
    template void TreeMortonCodes<u32>::build<f32_3>(
        sycl::queue &queue,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f32_3> &pos_buf);

    template void TreeMortonCodes<u32>::build<f32_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<f32_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u32>> TreeMortonCodes<u32>::build_raw<f32_3>(
        sycl::queue &queue,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f32_3> &pos_buf);

    // u32, u64_3
    template void TreeMortonCodes<u32>::build<u64_3>(
        sycl::queue &queue,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u64_3> &pos_buf);

    template void TreeMortonCodes<u32>::build<u64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<u64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u32>> TreeMortonCodes<u32>::build_raw<u64_3>(
        sycl::queue &queue,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u64_3> &pos_buf);

    // u64, f64_3
    template void TreeMortonCodes<u64>::build<f64_3>(
        sycl::queue &queue,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f64_3> &pos_buf);

    template void TreeMortonCodes<u64>::build<f64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<f64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u64>> TreeMortonCodes<u64>::build_raw<f64_3>(
        sycl::queue &queue,
        shammath::CoordRange<f64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f64_3> &pos_buf);

    // u64, f32_3
    template void TreeMortonCodes<u64>::build<f32_3>(
        sycl::queue &queue,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f32_3> &pos_buf);

    template void TreeMortonCodes<u64>::build<f32_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<f32_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u64>> TreeMortonCodes<u64>::build_raw<f32_3>(
        sycl::queue &queue,
        shammath::CoordRange<f32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<f32_3> &pos_buf);

    // u64, u64_3
    template void TreeMortonCodes<u64>::build<u64_3>(
        sycl::queue &queue,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u64_3> &pos_buf);

    template void TreeMortonCodes<u64>::build<u64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<u64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u64>> TreeMortonCodes<u64>::build_raw<u64_3>(
        sycl::queue &queue,
        shammath::CoordRange<u64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u64_3> &pos_buf);

    // u64, u32_3
    template void TreeMortonCodes<u64>::build<u32_3>(
        sycl::queue &queue,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u32_3> &pos_buf);

    template void TreeMortonCodes<u64>::build<u32_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<u32_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u64>> TreeMortonCodes<u64>::build_raw<u32_3>(
        sycl::queue &queue,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u32_3> &pos_buf);

    // u64, i64_3
    template void TreeMortonCodes<u64>::build<i64_3>(
        sycl::queue &queue,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<i64_3> &pos_buf);

    template void TreeMortonCodes<u64>::build<i64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<i64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u64>> TreeMortonCodes<u64>::build_raw<i64_3>(
        sycl::queue &queue,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<i64_3> &pos_buf);

    // u32, i64_3
    template void TreeMortonCodes<u32>::build<i64_3>(
        sycl::queue &queue,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<i64_3> &pos_buf);

    template void TreeMortonCodes<u32>::build<i64_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<i64_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u32>> TreeMortonCodes<u32>::build_raw<i64_3>(
        sycl::queue &queue,
        shammath::CoordRange<i64_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<i64_3> &pos_buf);

    // u32, u32_3
    template void TreeMortonCodes<u32>::build<u32_3>(
        sycl::queue &queue,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u32_3> &pos_buf);

    template void TreeMortonCodes<u32>::build<u32_3>(
        sham::DeviceScheduler_ptr dev_sched,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sham::DeviceBuffer<u32_3> &pos_buf);

    template std::unique_ptr<sycl::buffer<u32>> TreeMortonCodes<u32>::build_raw<u32_3>(
        sycl::queue &queue,
        shammath::CoordRange<u32_3> coord_range,
        u32 obj_cnt,
        sycl::buffer<u32_3> &pos_buf);

} // namespace shamrock::tree
