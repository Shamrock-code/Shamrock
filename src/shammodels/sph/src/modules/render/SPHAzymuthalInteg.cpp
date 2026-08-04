// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHAzymuthalInteg.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/SPHAzymuthalInteg.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <cmath>
#include <limits>

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays_buf = edges.ring_rays.value;
    sham::DeviceBuffer<T> &output_buf = edges.interpolated_field.value;

    u32 nring_rays = ring_rays_buf.get_size();
    if (output_buf.get_size() != nring_rays) {
        output_buf.resize_discard_data(nring_rays);
    }
    output_buf.fill(sham::VectorProperties<T>::get_zero());

    using u_morton = u32;
    using RTree    = RadixTree<u_morton, Tvec>;

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

        RTree tree(
            dev_sched, {aabb.lower, aabb.upper}, pos.get_buf(), obj_cnt, tree_reduction_level);

        tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
        tree.convert_bounding_box(shamsys::instance::get_compute_queue());

        auto &hpart_span = edges.h_part.get_spans().get(id);
        auto &field_span = edges.field_data.get_spans().get(id);
        auto &buf_hpart  = hpart_span.field_ref.get_buf();
        auto &buf_field  = field_span.field_ref.get_buf();

        RadixTreeField<Tscal> hmax_tree
            = tree.compute_int_boxes(shamsys::instance::get_compute_queue(), buf_hpart, 1);

        sham::EventList depends_list;
        T *render_field = output_buf.get_write_access(depends_list);

        const shammath::RingRay<Tvec> *ring_rays_ptr = ring_rays_buf.get_read_access(depends_list);

        auto xyz      = pos.get_buf().get_read_access(depends_list);
        auto hpart    = buf_hpart.get_read_access(depends_list);
        auto torender = buf_field.get_read_access(depends_list);

        sycl::event e2 = queue.submit(depends_list, [&, render_field](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator particle_looper(tree, cgh);

            sycl::accessor hmax{
                shambase::get_check_ref(hmax_tree.radix_tree_field_buf), cgh, sycl::read_only};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parallel_for(cgh, nring_rays, "compute azymuthal render", [=](u32 gid) {
                T acc = sham::VectorProperties<T>::get_zero();

                shammath::RingRay<Tvec> ring_ray = ring_rays_ptr[gid];
                Tvec ez                          = ring_ray.get_ez();

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin_cell, Tvec bmax_cell) -> bool {
                        Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                        auto interbox
                            = shammath::AABB<Tvec>{bmin_cell, bmax_cell}.expand_all(rint_cell);

                        return interbox.intersect_ring_ray_approx(ring_ray);
                    },
                    [&](u32 id_b) {
                        Tvec r_center = ring_ray.center - xyz[id_b];

                        Tscal z_val = sycl::dot(r_center, ez);
                        Tscal x_val = sycl::dot(r_center, ring_ray.e_x);
                        Tscal y_val = sycl::dot(r_center, ring_ray.e_y);
                        Tscal r_val = sycl::sqrt(x_val * x_val + y_val * y_val);

                        Tscal delta_r = r_val - ring_ray.radius;

                        Tscal rab2_ring = z_val * z_val + delta_r * delta_r;
                        Tscal h_b       = hpart[id_b];

                        if (rab2_ring > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2_ring);

                        T val = torender[id_b];

                        Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                        // TODO: account for curvature
                        acc += partmass * val * Kernel::Y_3d(rab, h_b, 4) / rho_b;
                    });

                render_field[gid] += acc;
            });
        });

        pos.get_buf().complete_event_state(e2);
        buf_hpart.complete_event_state(e2);
        buf_field.complete_event_state(e2);
        output_buf.complete_event_state(e2);
        ring_rays_buf.complete_event_state(e2);
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C6>;
