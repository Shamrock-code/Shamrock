// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

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

        sham::DeviceBuffer<shammath::Ray<Tvec>> ret{
            nx * ny, shamsys::instance::get_compute_scheduler_ptr()};

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tvec e_z = sycl::cross(delta_x, delta_y);
        e_z /= sycl::length(e_z);

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
        std::string field_name, Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny)
        -> sham::DeviceBuffer<Tfield> {

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format(
                    "compute_slice field_name: {}, center: {}, delta_x: {}, delta_y: {}, nx: {}, "
                    "ny: {}",
                    field_name,
                    center,
                    delta_x,
                    delta_y,
                    nx,
                    ny));
        }

        shambase::Timer t;
        t.start();

        auto ret = RenderFieldGetter<Tvec, Tfield, SPHKernel>(context, solver_config, storage)
                       .runner_function(
                           field_name, [&](auto field_getter) -> sham::DeviceBuffer<Tfield> {
                               return compute_slice(field_getter, center, delta_x, delta_y, nx, ny);
                           });

        t.end();
        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format("compute_slice took {}", t.get_time_str()));
        }

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::string field_name, Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny)
        -> sham::DeviceBuffer<Tfield> {

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format(
                    "compute_column_integ field_name: {}, center: {}, delta_x: {}, delta_y: {}, "
                    "nx: {}, "
                    "ny: {}",
                    field_name,
                    center,
                    delta_x,
                    delta_y,
                    nx,
                    ny));
        }

        shambase::Timer t;
        t.start();

        auto ret = RenderFieldGetter<Tvec, Tfield, SPHKernel>(context, solver_config, storage)
                       .runner_function(
                           field_name, [&](auto field_getter) -> sham::DeviceBuffer<Tfield> {
                               return compute_column_integ(
                                   field_getter, center, delta_x, delta_y, nx, ny);
                           });

        t.end();
        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "sph::CartesianRender",
                shambase::format("compute_column_integ took {}", t.get_time_str()));
        }

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

        sham::DeviceBuffer<Tfield> ret{nx * ny, shamsys::instance::get_compute_scheduler_ptr()};
        ret.fill(sham::VectorProperties<Tfield>::get_zero());

        auto positions = pixel_to_positions(center, delta_x, delta_y, nx, ny);

        using u_morton = u32;
        using RTree    = RadixTree<u_morton, Tvec>;

        shamrock::patch::PatchCoordTransform<Tvec> transf
            = scheduler().get_sim_box().template get_patch_transform<Tvec>();

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch cur_p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shammath::CoordRange<Tvec> box = transf.to_obj_coord(cur_p);

            PatchDataField<Tvec> &main_field = pdat.get_field<Tvec>(0);

            auto &buf_xyz = pdat.get_field<Tvec>(0).get_buf();
            auto &buf_hpart
                = pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf();

            auto &buf_field_to_render = field_getter(cur_p, pdat);

            u32 obj_cnt = main_field.get_obj_cnt();

            RTree tree(
                shamsys::instance::get_compute_scheduler_ptr(),
                {box.lower, box.upper},
                buf_xyz,
                obj_cnt,
                solver_config.tree_reduction_level);

            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());

            RadixTreeField<Tscal> hmax_tree = tree.compute_int_boxes(
                shamsys::instance::get_compute_queue(),
                pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf(),
                1);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::EventList depends_list;
            Tfield *render_field = ret.get_write_access(depends_list);

            const Tvec *pixel_positions = positions.get_read_access(depends_list);

            auto xyz      = buf_xyz.get_read_access(depends_list);
            auto hpart    = buf_hpart.get_read_access(depends_list);
            auto torender = buf_field_to_render.get_read_access(depends_list);

            sycl::event e2 = q.submit(depends_list, [&, render_field](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator particle_looper(tree, cgh);

                sycl::accessor hmax{
                    shambase::get_check_ref(hmax_tree.radix_tree_field_buf), cgh, sycl::read_only};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal partmass = solver_config.gpart_mass;

                shambase::parallel_for(cgh, nx * ny, "compute slice render", [=](u32 gid) {
                    Tvec pos_render = pixel_positions[gid];

                    Tfield ret = sham::VectorProperties<Tfield>::get_zero();

                    particle_looper.rtree_for(
                        [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                            Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                            auto interbox
                                = shammath::CoordRange<Tvec>{bmin, bmax}.expand_all(rint_cell);

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

                            Tfield val = torender[id_b];

                            Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                            ret += partmass * val * Kernel::W_3d(rab, h_b) / rho_b;
                        });

                    render_field[gid] += ret;
                });
            });

            buf_xyz.complete_event_state(e2);
            buf_hpart.complete_event_state(e2);
            buf_field_to_render.complete_event_state(e2);
            ret.complete_event_state(e2);
            positions.complete_event_state(e2);
        });

        shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

        return ret;
    }
    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_column_integ(
        std::function<field_getter_t> field_getter,
        Tvec center,
        Tvec delta_x,
        Tvec delta_y,
        u32 nx,
        u32 ny) -> sham::DeviceBuffer<Tfield> {

        sham::DeviceBuffer<Tfield> ret{nx * ny, shamsys::instance::get_compute_scheduler_ptr()};
        ret.fill(sham::VectorProperties<Tfield>::get_zero());

        auto rays = pixel_to_orthographic_rays(center, delta_x, delta_y, nx, ny);

        using u_morton = u32;
        using RTree    = RadixTree<u_morton, Tvec>;

        shamrock::patch::PatchCoordTransform<Tvec> transf
            = scheduler().get_sim_box().template get_patch_transform<Tvec>();

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch cur_p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shammath::CoordRange<Tvec> box = transf.to_obj_coord(cur_p);

            PatchDataField<Tvec> &main_field = pdat.get_field<Tvec>(0);

            auto &buf_xyz = pdat.get_field<Tvec>(0).get_buf();
            auto &buf_hpart
                = pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf();

            auto &buf_field_to_render = field_getter(cur_p, pdat);

            u32 obj_cnt = main_field.get_obj_cnt();

            RTree tree(
                shamsys::instance::get_compute_scheduler_ptr(),
                {box.lower, box.upper},
                buf_xyz,
                obj_cnt,
                solver_config.tree_reduction_level);

            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());

            RadixTreeField<Tscal> hmax_tree = tree.compute_int_boxes(
                shamsys::instance::get_compute_queue(),
                pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf(),
                1);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::EventList depends_list;
            Tfield *render_field = ret.get_write_access(depends_list);

            const shammath::Ray<Tvec> *image_rays = rays.get_read_access(depends_list);

            auto xyz      = buf_xyz.get_read_access(depends_list);
            auto hpart    = buf_hpart.get_read_access(depends_list);
            auto torender = buf_field_to_render.get_read_access(depends_list);

            sycl::event e2 = q.submit(depends_list, [&, render_field](sycl::handler &cgh) {
                shamrock::tree::ObjectIterator particle_looper(tree, cgh);

                sycl::accessor hmax{
                    shambase::get_check_ref(hmax_tree.radix_tree_field_buf), cgh, sycl::read_only};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal partmass = solver_config.gpart_mass;

                shambase::parallel_for(cgh, nx * ny, "compute slice render", [=](u32 gid) {
                    u32 ix          = gid % nx;
                    u32 iy          = gid / nx;
                    f64 fx          = ((f64(ix) + 0.5) / nx) - 0.5;
                    f64 fy          = ((f64(iy) + 0.5) / ny) - 0.5;
                    Tvec pos_render = center + delta_x * fx + delta_y * fy;

                    Tfield ret = sham::VectorProperties<Tfield>::get_zero();

                    shammath::Ray<Tvec> ray = image_rays[gid];

                    particle_looper.rtree_for(
                        [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                            Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                            auto interbox = shammath::AABB<Tvec>{bmin, bmax}.expand_all(rint_cell);

                            return interbox.intersect_ray(ray);
                        },
                        [&](u32 id_b) {
                            Tvec dr = pos_render - xyz[id_b];

                            dr -= ray.direction * sycl::dot(dr, ray.direction);

                            Tscal rab2 = sycl::dot(dr, dr);
                            Tscal h_b  = hpart[id_b];

                            if (rab2 > h_b * h_b * Rker2) {
                                return;
                            }

                            Tscal rab = sycl::sqrt(rab2);

                            Tfield val = torender[id_b];

                            Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                            ret += partmass * val * Kernel::Y_3d(rab, h_b, 4) / rho_b;
                        });

                    render_field[gid] += ret;
                });
            });

            buf_xyz.complete_event_state(e2);
            buf_hpart.complete_event_state(e2);
            buf_field_to_render.complete_event_state(e2);
            ret.complete_event_state(e2);
            rays.complete_event_state(e2);
        });

        shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

        return ret;
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
