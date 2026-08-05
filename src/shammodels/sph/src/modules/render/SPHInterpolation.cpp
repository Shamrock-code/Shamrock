// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHInterpolation.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/SPHInterpolation.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <cmath>
#include <limits>

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<Tvec> &interp_points_buf = edges.interp_points.value;
    sham::DeviceBuffer<T> &output_buf                 = edges.interpolated_field.value;

    u32 npoints = interp_points_buf.get_size();
    if (output_buf.get_size() != npoints) {
        output_buf.resize_discard_data(npoints);
    }
    output_buf.fill(sham::VectorProperties<T>::get_zero());

    using u_morton = u32;
    using Tree     = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    Tscal partmass           = edges.gpart_mass.data;
    u32 tree_reduction_level = edges.tree_reduction_level.data;
    sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched           = shamsys::instance::get_compute_scheduler_ptr();

    part_counts.for_each([&](u64 id, u32 count) {
        if (count == 0) {
            return;
        }

        PatchDataField<Tvec> &pos = edges.positions.get_field(id);
        if (pos.is_empty()) {
            return;
        }

        Tvec bmax = pos.compute_max();
        Tvec bmin = pos.compute_min();

        shammath::AABB<Tvec> aabb(bmin, bmax);

        Tscal infty = std::numeric_limits<Tscal>::infinity();

        aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
        aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
        aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
        aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
        aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
        aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

        u32 obj_cnt = pos.get_obj_cnt();

        Tree tree = Tree::make_empty(dev_sched);
        tree.rebuild_from_positions(pos.get_buf(), obj_cnt, aabb, tree_reduction_level);

        auto &hpart_span = edges.h_part.get_spans().get(id);
        auto &field_span = edges.field_data.get_spans().get(id);
        auto &buf_hpart  = hpart_span.field_ref.get_buf();
        auto &buf_field  = field_span.field_ref.get_buf();

        auto hmax_tree = shamtree::compute_tree_field_max_field<Tscal>(
            tree.structure,
            tree.reduced_morton_set.get_leaf_cell_iterator(),
            shamtree::new_empty_karras_radix_tree_field<Tscal>(),
            buf_hpart);

        auto obj_it = tree.get_object_iterator();

        sham::kernel_call(
            queue,
            sham::MultiRef{
                interp_points_buf,
                pos.get_buf(),
                buf_hpart,
                buf_field,
                obj_it,
                hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            npoints,
            [=](u32 gid,
                const Tvec *__restrict pixel_positions,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                Tvec pos_render = pixel_positions[gid];

                T acc = sham::VectorProperties<T>::get_zero();

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                        return node_aabb.expand_all(rint_cell).contains_asymmetric(pos_render);
                    },
                    [&](u32 id_b) {
                        Tvec dr    = pos_render - xyz[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        T val = torender[id_b];

                        Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                        acc += partmass * val * Kernel::W_3d(rab, h_b) / rho_b;
                    });

                render_field[gid] += acc;
            });
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C6>;
