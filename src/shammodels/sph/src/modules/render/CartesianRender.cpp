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
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shammodels/sph/modules/render/SPHAzymuthalInteg.hpp"
#include "shammodels/sph/modules/render/SPHColumnInteg.hpp"
#include "shammodels/sph/modules/render/SPHInterpolation.hpp"
#include "shamrock/solvergraph/DeviceBufferEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"

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
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
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
                sham::format(
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
                "sph::CartesianRender", sham::format("compute_slice took {}", t.get_time_str()));
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
                sham::format(
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
                sham::format("compute_column_integ took {}", t.get_time_str()));
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
                sham::format(
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
                sham::format("compute_azymuthal_integ took {}", t.get_time_str()));
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
