// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "RadixTreeMortonBuilder.hpp"
#include "kernels/key_morton_sort.hpp"
#include "shamalgs/algorithm.hpp"
#include "shambase/exception.hpp"
#include "shambase/integer_sycl.hpp"
#include "shamrock/sfc/MortonKernels.hpp"
#include "shamrock/sfc/morton.hpp"
#include "shamsys/legacy/log.hpp"

template<class morton_t, class pos_t, u32 dim>
void RadixTreeMortonBuilder<morton_t, pos_t, dim>::build(
    sycl::queue &queue,
    std::tuple<pos_t, pos_t> bounding_box,
    sycl::buffer<pos_t> &pos_buf,
    u32 cnt_obj,
    std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &out_buf_particle_index_map) {

    using namespace logger;
    using namespace shamrock::sfc;

    if (cnt_obj > i32_max - 1) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "number of element in patch above i32_max-1");
    }

    debug_sycl_ln("RadixTree", "box dim :", bounding_box);

    u32 morton_len = shambase::roundup_pow2_clz(cnt_obj);

    debug_sycl_ln("RadixTree", "morton buffer lenght :", morton_len);
    out_buf_morton = std::make_unique<sycl::buffer<morton_t>>(morton_len);

    MortonKernels<morton_t, pos_t, dim>::sycl_xyz_to_morton(queue,
                                                            cnt_obj,
                                                            pos_buf,
                                                            std::get<0>(bounding_box),
                                                            std::get<1>(bounding_box),
                                                            out_buf_morton);

    MortonKernels<morton_t, pos_t, dim>::sycl_fill_trailling_buffer(
        queue, cnt_obj, morton_len, out_buf_morton);

    out_buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(
        shamalgs::algorithm::gen_buffer_index(queue, morton_len));

    sycl_sort_morton_key_pair(queue, morton_len, out_buf_particle_index_map, out_buf_morton);
}

template<class morton_t, class pos_t, u32 dim>
void RadixTreeMortonBuilder<morton_t, pos_t, dim>::build_raw(
    sycl::queue &queue,
    std::tuple<pos_t, pos_t> bounding_box,
    sycl::buffer<pos_t> &pos_buf,
    u32 cnt_obj,
    std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton) {

    using namespace logger;
    using namespace shamrock::sfc;

    if (cnt_obj > i32_max - 1) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "number of element in patch above i32_max-1");
    }

    debug_sycl_ln("RadixTree", "box dim :", bounding_box);

    debug_sycl_ln("RadixTree", "morton buffer lenght :", cnt_obj);
    out_buf_morton = std::make_unique<sycl::buffer<morton_t>>(cnt_obj);

    MortonKernels<morton_t, pos_t, dim>::sycl_xyz_to_morton(queue,
                                                            cnt_obj,
                                                            pos_buf,
                                                            std::get<0>(bounding_box),
                                                            std::get<1>(bounding_box),
                                                            out_buf_morton);
}

template class RadixTreeMortonBuilder<u32, f32_3, 3>;
template class RadixTreeMortonBuilder<u64, f32_3, 3>;
template class RadixTreeMortonBuilder<u32, f64_3, 3>;
template class RadixTreeMortonBuilder<u64, f64_3, 3>;
template class RadixTreeMortonBuilder<u32, u32_3, 3>;
template class RadixTreeMortonBuilder<u64, u32_3, 3>;
template class RadixTreeMortonBuilder<u32, u64_3, 3>;
template class RadixTreeMortonBuilder<u64, u64_3, 3>;
template class RadixTreeMortonBuilder<u32, i64_3, 3>;
template class RadixTreeMortonBuilder<u64, i64_3, 3>;

/////
