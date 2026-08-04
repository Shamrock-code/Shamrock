// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CartesianRender.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <cmath>
#include <limits>

namespace shamrock::solvergraph {

    template<class T>
    class DeviceBufferEdge : public IEdgeNamed {

        public:
        sham::DeviceBuffer<T> value;

        DeviceBufferEdge(std::string name, std::string texsymbol)
            : IEdgeNamed(std::move(name), std::move(texsymbol)),
              value(0, shamsys::instance::get_compute_scheduler_ptr()) {}

        inline virtual void free_alloc() override { value.resize(0); }
    };

} // namespace shamrock::solvergraph

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, tree_reduction_level)                              \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, h_part)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, field_data)                                         \
    X_RO(shamrock::solvergraph::DeviceBufferEdge<Tvec>, interp_points)                             \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::DeviceBufferEdge<T>, interpolated_field)

namespace shammodels::sph::modules {

    template<class Tvec, class T, template<class> class SPHKernel>
    class SPHInterpolation : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() override {

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
                    dev_sched,
                    {aabb.lower, aabb.upper},
                    pos.get_buf(),
                    obj_cnt,
                    tree_reduction_level);

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

                const Tvec *pixel_positions = interp_points_buf.get_read_access(depends_list);

                auto xyz      = pos.get_buf().get_read_access(depends_list);
                auto hpart    = buf_hpart.get_read_access(depends_list);
                auto torender = buf_field.get_read_access(depends_list);

                sycl::event e2 = queue.submit(depends_list, [&, render_field](sycl::handler &cgh) {
                    shamrock::tree::ObjectIterator particle_looper(tree, cgh);

                    sycl::accessor hmax{
                        shambase::get_check_ref(hmax_tree.radix_tree_field_buf),
                        cgh,
                        sycl::read_only};

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    shambase::parallel_for(cgh, npoints, "compute slice render", [=](u32 gid) {
                        Tvec pos_render = pixel_positions[gid];

                        T acc = sham::VectorProperties<T>::get_zero();

                        particle_looper.rtree_for(
                            [&](u32 node_id, Tvec bmin_cell, Tvec bmax_cell) -> bool {
                                Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                                auto interbox
                                    = shammath::CoordRange<Tvec>{bmin_cell, bmax_cell}.expand_all(
                                        rint_cell);

                                return interbox.contain_pos(pos_render);
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

                pos.get_buf().complete_event_state(e2);
                buf_hpart.complete_event_state(e2);
                buf_field.complete_event_state(e2);
                output_buf.complete_event_state(e2);
                interp_points_buf.complete_event_state(e2);
            });

            shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
        }

        inline std::string _impl_get_label() const override { return "SPHInterpolation"; }

        inline std::string _impl_get_tex() const override { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, tree_reduction_level)                              \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, h_part)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, field_data)                                         \
    X_RO(shamrock::solvergraph::DeviceBufferEdge<shammath::Ray<Tvec>>, rays)                       \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::DeviceBufferEdge<T>, interpolated_field)

namespace shammodels::sph::modules {

    template<class Tvec, class T, template<class> class SPHKernel>
    class SPHColumnInteg : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() override {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            edges.positions.check_sizes(part_counts);
            edges.h_part.check_sizes(part_counts);
            edges.field_data.check_sizes(part_counts);

            const sham::DeviceBuffer<shammath::Ray<Tvec>> &rays_buf = edges.rays.value;
            sham::DeviceBuffer<T> &output_buf = edges.interpolated_field.value;

            u32 nrays = rays_buf.get_size();
            if (output_buf.get_size() != nrays) {
                output_buf.resize_discard_data(nrays);
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
                    dev_sched,
                    {aabb.lower, aabb.upper},
                    pos.get_buf(),
                    obj_cnt,
                    tree_reduction_level);

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

                const shammath::Ray<Tvec> *image_rays = rays_buf.get_read_access(depends_list);

                auto xyz      = pos.get_buf().get_read_access(depends_list);
                auto hpart    = buf_hpart.get_read_access(depends_list);
                auto torender = buf_field.get_read_access(depends_list);

                sycl::event e2 = queue.submit(depends_list, [&, render_field](sycl::handler &cgh) {
                    shamrock::tree::ObjectIterator particle_looper(tree, cgh);

                    sycl::accessor hmax{
                        shambase::get_check_ref(hmax_tree.radix_tree_field_buf),
                        cgh,
                        sycl::read_only};

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    shambase::parallel_for(cgh, nrays, "compute column render", [=](u32 gid) {
                        T acc = sham::VectorProperties<T>::get_zero();

                        shammath::Ray<Tvec> ray = image_rays[gid];

                        particle_looper.rtree_for(
                            [&](u32 node_id, Tvec bmin_cell, Tvec bmax_cell) -> bool {
                                Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                                auto interbox
                                    = shammath::AABB<Tvec>{bmin_cell, bmax_cell}.expand_all(
                                        rint_cell);

                                return interbox.intersect_ray(ray);
                            },
                            [&](u32 id_b) {
                                Tvec dr = ray.origin - xyz[id_b];

                                dr -= ray.direction * sycl::dot(dr, ray.direction);

                                Tscal rab2 = sycl::dot(dr, dr);
                                Tscal h_b  = hpart[id_b];

                                if (rab2 > h_b * h_b * Rker2) {
                                    return;
                                }

                                Tscal rab = sycl::sqrt(rab2);

                                T val = torender[id_b];

                                Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                                acc += partmass * val * Kernel::Y_3d(rab, h_b, 4) / rho_b;
                            });

                        render_field[gid] += acc;
                    });
                });

                pos.get_buf().complete_event_state(e2);
                buf_hpart.complete_event_state(e2);
                buf_field.complete_event_state(e2);
                output_buf.complete_event_state(e2);
                rays_buf.complete_event_state(e2);
            });

            shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
        }

        inline std::string _impl_get_label() const override { return "SPHColumnInteg"; }

        inline std::string _impl_get_tex() const override { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, tree_reduction_level)                              \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, h_part)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, field_data)                                         \
    X_RO(shamrock::solvergraph::DeviceBufferEdge<shammath::RingRay<Tvec>>, ring_rays)              \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::DeviceBufferEdge<T>, interpolated_field)

namespace shammodels::sph::modules {

    template<class Tvec, class T, template<class> class SPHKernel>
    class SPHAzymuthalInteg : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() override {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            edges.positions.check_sizes(part_counts);
            edges.h_part.check_sizes(part_counts);
            edges.field_data.check_sizes(part_counts);

            const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays_buf
                = edges.ring_rays.value;
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
                    dev_sched,
                    {aabb.lower, aabb.upper},
                    pos.get_buf(),
                    obj_cnt,
                    tree_reduction_level);

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

                const shammath::RingRay<Tvec> *ring_rays_ptr
                    = ring_rays_buf.get_read_access(depends_list);

                auto xyz      = pos.get_buf().get_read_access(depends_list);
                auto hpart    = buf_hpart.get_read_access(depends_list);
                auto torender = buf_field.get_read_access(depends_list);

                sycl::event e2 = queue.submit(depends_list, [&, render_field](sycl::handler &cgh) {
                    shamrock::tree::ObjectIterator particle_looper(tree, cgh);

                    sycl::accessor hmax{
                        shambase::get_check_ref(hmax_tree.radix_tree_field_buf),
                        cgh,
                        sycl::read_only};

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    shambase::parallel_for(
                        cgh, nring_rays, "compute azymuthal render", [=](u32 gid) {
                            T acc = sham::VectorProperties<T>::get_zero();

                            shammath::RingRay<Tvec> ring_ray = ring_rays_ptr[gid];
                            Tvec ez                          = ring_ray.get_ez();

                            particle_looper.rtree_for(
                                [&](u32 node_id, Tvec bmin_cell, Tvec bmax_cell) -> bool {
                                    Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                                    auto interbox
                                        = shammath::AABB<Tvec>{bmin_cell, bmax_cell}.expand_all(
                                            rint_cell);

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

                                    Tscal rho_b
                                        = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

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

        inline std::string _impl_get_label() const override { return "SPHAzymuthalInteg"; }

        inline std::string _impl_get_tex() const override { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

namespace shammodels::sph::modules {

    template<class Tvec>
    sham::DeviceBuffer<Tvec> pixel_to_positions(
        Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny) {

        sham::DeviceBuffer<Tvec> ret{nx * ny, shamsys::instance::get_compute_scheduler_ptr()};

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q, sham::MultiRef{}, sham::MultiRef{ret}, nx * ny, [=](u32 gid, Tvec *position) {
                u32 ix        = gid % nx;
                u32 iy        = gid / nx;
                f64 fx        = ((f64(ix) + 0.5) / nx) - 0.5;
                f64 fy        = ((f64(iy) + 0.5) / ny) - 0.5;
                position[gid] = center + delta_x * fx + delta_y * fy;
            });

        return ret;
    }

    template<class Tvec>
    sham::DeviceBuffer<shammath::Ray<Tvec>> pixel_to_orthographic_rays(
        Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny) {

        using Tscal = shambase::VecComponent<Tvec>;

        sham::DeviceBuffer<shammath::Ray<Tvec>> ret{
            nx * ny, shamsys::instance::get_compute_scheduler_ptr()};

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tvec e_z  = sycl::cross(delta_x, delta_y);
        Tscal len = sycl::length(e_z);
        if (!(len > 0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "The cross product of delta_x and delta_y is zero\n"
                "  args :"
                "    center  = {}\n"
                "    delta_x = {}\n"
                "    delta_y = {}\n"
                "    nx      = {}\n"
                "    ny      = {}\n"
                "  -> e_z = {}\n",
                center,
                delta_x,
                delta_y,
                nx,
                ny,
                e_z));
        }
        e_z /= len;

        sham::kernel_call(
            q,
            sham::MultiRef{},
            sham::MultiRef{ret},
            nx * ny,
            [=](u32 gid, shammath::Ray<Tvec> *ray) {
                u32 ix          = gid % nx;
                u32 iy          = gid / nx;
                f64 fx          = ((f64(ix) + 0.5) / nx) - 0.5;
                f64 fy          = ((f64(iy) + 0.5) / ny) - 0.5;
                Tvec pos_render = center + delta_x * fx + delta_y * fy;

                ray[gid] = shammath::Ray<Tvec>(pos_render, e_z);
            });

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_slice(
        std::string field_name,
        const sham::DeviceBuffer<Tvec> &positions,
        std::optional<std::function<py::array_t<Tfield>(size_t, pybind11::dict &)>> custom_getter)
        -> sham::DeviceBuffer<Tfield> {

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format(
                    "compute_slice field_name: {}, positions count: {}",
                    field_name,
                    positions.get_size()));
        }

        shambase::Timer t;
        t.start();

        auto ret = RenderFieldGetter<Tvec, Tfield, SPHKernel>(context, solver_config, storage)
                       .runner_function(
                           field_name,
                           [&](auto field_getter) -> sham::DeviceBuffer<Tfield> {
                               return compute_slice(field_getter, positions);
                           },
                           custom_getter);

        t.stop();
        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format("compute_slice took {}", t.get_time_str()));
        }

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::string field_name,
        const sham::DeviceBuffer<shammath::Ray<Tvec>> &rays,
        std::optional<std::function<py::array_t<Tfield>(size_t, pybind11::dict &)>> custom_getter)
        -> sham::DeviceBuffer<Tfield> {

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format(
                    "compute_column_integ field_name: {}, rays count: {}",
                    field_name,
                    rays.get_size()));
        }

        shambase::Timer t;
        t.start();

        auto ret = RenderFieldGetter<Tvec, Tfield, SPHKernel>(context, solver_config, storage)
                       .runner_function(
                           field_name,
                           [&](auto field_getter) -> sham::DeviceBuffer<Tfield> {
                               return compute_column_integ(field_getter, rays);
                           },
                           custom_getter);

        t.stop();
        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format("compute_column_integ took {}", t.get_time_str()));
        }

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_azymuthal_integ(
        std::string field_name,
        const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays,
        std::optional<std::function<py::array_t<Tfield>(size_t, pybind11::dict &)>> custom_getter)
        -> sham::DeviceBuffer<Tfield> {

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format(
                    "compute_azymuthal_integ field_name: {}, ring_rays count: {}",
                    field_name,
                    ring_rays.get_size()));
        }

        shambase::Timer t;
        t.start();

        auto ret = RenderFieldGetter<Tvec, Tfield, SPHKernel>(context, solver_config, storage)
                       .runner_function(
                           field_name,
                           [&](auto field_getter) -> sham::DeviceBuffer<Tfield> {
                               return compute_azymuthal_integ(field_getter, ring_rays);
                           },
                           custom_getter);

        t.stop();
        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format("compute_azymuthal_integ took {}", t.get_time_str()));
        }

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_slice(
        std::function<field_getter_t> field_getter, const sham::DeviceBuffer<Tvec> &positions)
        -> sham::DeviceBuffer<Tfield> {

        auto part_counts = shamrock::solvergraph::Indexes<u32>::make_shared("part_counts", "N");
        auto positions_refs
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("positions", "\\mathbf{r}");
        auto hpart_refs = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_part", "h");
        auto field_data
            = std::make_shared<shamrock::solvergraph::Field<Tfield>>(1, "field_data", "f");

        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_dd;
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> h_dd;

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                u64 id  = cur_p.id_patch;
                u32 cnt = pdat.get_obj_cnt();

                part_counts->indexes.add_obj(id, std::move(cnt));
                pos_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(0)));
                h_dd.add_obj(
                    id, std::ref(pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart"))));
            });

        positions_refs->set_refs(pos_dd);
        hpart_refs->set_refs(h_dd);

        field_data->ensure_sizes(part_counts->indexes);

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                const sham::DeviceBuffer<Tfield> &src = field_getter(cur_p, pdat);
                field_data->get(cur_p.id_patch).overwrite(src, static_cast<u32>(src.get_size()));
            });

        auto gpart_mass  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("gpart_mass", "m");
        gpart_mass->data = solver_config.gpart_mass;

        auto tree_reduction_level
            = shamrock::solvergraph::IDataEdge<u32>::make_shared("tree_reduction_level", "l");
        tree_reduction_level->data = solver_config.tree_reduction_level;

        auto interp_points = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<Tvec>>(
            "interp_points", "\\mathbf{q}");
        interp_points->value.resize(positions.get_size());
        interp_points->value.copy_from(positions);

        auto interpolated_field = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<Tfield>>(
            "interpolated_field", "f_{\\rm interp}");

        auto node = std::make_shared<SPHInterpolation<Tvec, Tfield, SPHKernel>>();
        node->set_edges(
            gpart_mass,
            tree_reduction_level,
            part_counts,
            positions_refs,
            hpart_refs,
            field_data,
            interp_points,
            interpolated_field);
        node->evaluate();

        sham::DeviceBuffer<Tfield> ret{
            interpolated_field->value.get_size(), shamsys::instance::get_compute_scheduler_ptr()};
        ret.copy_from(interpolated_field->value);

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::function<field_getter_t> field_getter,
        const sham::DeviceBuffer<shammath::Ray<Tvec>> &rays) -> sham::DeviceBuffer<Tfield> {

        auto part_counts = shamrock::solvergraph::Indexes<u32>::make_shared("part_counts", "N");
        auto positions_refs
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("positions", "\\mathbf{r}");
        auto hpart_refs = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_part", "h");
        auto field_data
            = std::make_shared<shamrock::solvergraph::Field<Tfield>>(1, "field_data", "f");

        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_dd;
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> h_dd;

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                u64 id  = cur_p.id_patch;
                u32 cnt = pdat.get_obj_cnt();

                part_counts->indexes.add_obj(id, std::move(cnt));
                pos_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(0)));
                h_dd.add_obj(
                    id, std::ref(pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart"))));
            });

        positions_refs->set_refs(pos_dd);
        hpart_refs->set_refs(h_dd);

        field_data->ensure_sizes(part_counts->indexes);

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                const sham::DeviceBuffer<Tfield> &src = field_getter(cur_p, pdat);
                field_data->get(cur_p.id_patch).overwrite(src, static_cast<u32>(src.get_size()));
            });

        auto gpart_mass  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("gpart_mass", "m");
        gpart_mass->data = solver_config.gpart_mass;

        auto tree_reduction_level
            = shamrock::solvergraph::IDataEdge<u32>::make_shared("tree_reduction_level", "l");
        tree_reduction_level->data = solver_config.tree_reduction_level;

        auto rays_edge
            = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<shammath::Ray<Tvec>>>(
                "rays", "\\mathbf{r}_{\\rm ray}");
        rays_edge->value.resize(rays.get_size());
        rays_edge->value.copy_from(rays);

        auto interpolated_field = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<Tfield>>(
            "interpolated_field", "f_{\\rm interp}");

        auto node = std::make_shared<SPHColumnInteg<Tvec, Tfield, SPHKernel>>();
        node->set_edges(
            gpart_mass,
            tree_reduction_level,
            part_counts,
            positions_refs,
            hpart_refs,
            field_data,
            rays_edge,
            interpolated_field);
        node->evaluate();

        sham::DeviceBuffer<Tfield> ret{
            interpolated_field->value.get_size(), shamsys::instance::get_compute_scheduler_ptr()};
        ret.copy_from(interpolated_field->value);

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_azymuthal_integ(
        std::function<field_getter_t> field_getter,
        const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays)
        -> sham::DeviceBuffer<Tfield> {

        auto part_counts = shamrock::solvergraph::Indexes<u32>::make_shared("part_counts", "N");
        auto positions_refs
            = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("positions", "\\mathbf{r}");
        auto hpart_refs = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("h_part", "h");
        auto field_data
            = std::make_shared<shamrock::solvergraph::Field<Tfield>>(1, "field_data", "f");

        shamrock::solvergraph::DDPatchDataFieldRef<Tvec> pos_dd;
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> h_dd;

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                u64 id  = cur_p.id_patch;
                u32 cnt = pdat.get_obj_cnt();

                part_counts->indexes.add_obj(id, std::move(cnt));
                pos_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(0)));
                h_dd.add_obj(
                    id, std::ref(pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart"))));
            });

        positions_refs->set_refs(pos_dd);
        hpart_refs->set_refs(h_dd);

        field_data->ensure_sizes(part_counts->indexes);

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                const sham::DeviceBuffer<Tfield> &src = field_getter(cur_p, pdat);
                field_data->get(cur_p.id_patch).overwrite(src, static_cast<u32>(src.get_size()));
            });

        auto gpart_mass  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("gpart_mass", "m");
        gpart_mass->data = solver_config.gpart_mass;

        auto tree_reduction_level
            = shamrock::solvergraph::IDataEdge<u32>::make_shared("tree_reduction_level", "l");
        tree_reduction_level->data = solver_config.tree_reduction_level;

        auto ring_rays_edge
            = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<shammath::RingRay<Tvec>>>(
                "ring_rays", "\\mathbf{r}_{\\rm ring}");
        ring_rays_edge->value.resize(ring_rays.get_size());
        ring_rays_edge->value.copy_from(ring_rays);

        auto interpolated_field = std::make_shared<shamrock::solvergraph::DeviceBufferEdge<Tfield>>(
            "interpolated_field", "f_{\\rm interp}");

        auto node = std::make_shared<SPHAzymuthalInteg<Tvec, Tfield, SPHKernel>>();
        node->set_edges(
            gpart_mass,
            tree_reduction_level,
            part_counts,
            positions_refs,
            hpart_refs,
            field_data,
            ring_rays_edge,
            interpolated_field);
        node->evaluate();

        sham::DeviceBuffer<Tfield> ret{
            interpolated_field->value.get_size(), shamsys::instance::get_compute_scheduler_ptr()};
        ret.copy_from(interpolated_field->value);

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_slice(
        std::function<field_getter_t> field_getter,
        Tvec center,
        Tvec delta_x,
        Tvec delta_y,
        u32 nx,
        u32 ny) -> sham::DeviceBuffer<Tfield> {

        auto positions = pixel_to_positions(center, delta_x, delta_y, nx, ny);

        return compute_slice(field_getter, positions);
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::function<field_getter_t> field_getter,
        Tvec center,
        Tvec delta_x,
        Tvec delta_y,
        u32 nx,
        u32 ny) -> sham::DeviceBuffer<Tfield> {

        auto rays = pixel_to_orthographic_rays(center, delta_x, delta_y, nx, ny);

        return compute_column_integ(field_getter, rays);
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_slice(
        std::string field_name,
        Tvec center,
        Tvec delta_x,
        Tvec delta_y,
        u32 nx,
        u32 ny,
        std::optional<std::function<pybind11::array_t<Tfield>(size_t, pybind11::dict &)>>
            custom_getter) -> sham::DeviceBuffer<Tfield> {
        auto positions = pixel_to_positions(center, delta_x, delta_y, nx, ny);
        return compute_slice(field_name, positions, custom_getter);
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::string field_name,
        Tvec center,
        Tvec delta_x,
        Tvec delta_y,
        u32 nx,
        u32 ny,
        std::optional<std::function<pybind11::array_t<Tfield>(size_t, pybind11::dict &)>>
            custom_getter) -> sham::DeviceBuffer<Tfield> {
        auto rays = pixel_to_orthographic_rays(center, delta_x, delta_y, nx, ny);
        return compute_column_integ(field_name, rays, custom_getter);
    }

} // namespace shammodels::sph::modules

using namespace shammath;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M4>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M6>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M8>;

template class shammodels::sph::modules::CartesianRender<f64_3, f64, C2>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, C4>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, C6>;

template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, M4>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, M6>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, M8>;

template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, C2>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, C4>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64_3, C6>;
