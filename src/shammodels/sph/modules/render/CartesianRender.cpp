// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CartesianRender.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto CartesianRender<Tvec, Tfield, SPHKernel>::compute_slice(
        std::string field_name, Tvec center, Tvec delta_x, Tvec delta_y, u32 nx, u32 ny)
        -> sham::DeviceBuffer<Tfield> {

        if (field_name == "rho" && std::is_same_v<Tscal, Tfield>) {
            using namespace shamrock;
            using namespace shamrock::patch;
            shamrock::SchedulerUtility utility(scheduler());
            shamrock::ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
                logger::debug_ln("sph::vtk", "compute rho field for patch ", p.id_patch);
                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_h{
                        shambase::get_check_ref(
                            pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("hpart"))
                                .get_buf()),
                        cgh,
                        sycl::read_only};

                    sycl::accessor acc_rho{
                        shambase::get_check_ref(density.get_buf(p.id_patch)),
                        cgh,
                        sycl::write_only,
                        sycl::no_init};
                    const Tscal part_mass = solver_config.gpart_mass;

                    cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                        u32 gid = (u32) item.get_id();
                        using namespace shamrock::sph;
                        Tscal rho_ha = rho_h(part_mass, acc_h[gid], Kernel::hfactd);
                        acc_rho[gid] = rho_ha;
                    });
                });
            });

            auto field_source_getter
                = [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchData &pdat)
                -> const std::unique_ptr<sycl::buffer<Tfield>> & {
                return density.get_buf(cur_p.id_patch);
            };

            return compute_slice(field_source_getter, center, delta_x, delta_y, nx, ny);
        }

        auto field_source_getter =
            [&](const shamrock::patch::Patch cur_p,
                shamrock::patch::PatchData &pdat) -> const std::unique_ptr<sycl::buffer<Tfield>> & {
            return pdat.get_field<Tfield>(pdat.pdl.get_field_idx<Tfield>(field_name)).get_buf();
        };

        return compute_slice(field_source_getter, center, delta_x, delta_y, nx, ny);
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
        ret.fill({});

        using u_morton = u32;
        using RTree    = RadixTree<u_morton, Tvec>;

        shamrock::patch::PatchCoordTransform<Tvec> transf
            = scheduler().get_sim_box().template get_patch_transform<Tvec>();

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch cur_p,
                                                    shamrock::patch::PatchData &pdat) {
            shammath::CoordRange<Tvec> box = transf.to_obj_coord(cur_p);

            PatchDataField<Tvec> &main_field = pdat.get_field<Tvec>(0);

            auto &buf_xyz = pdat.get_field<Tvec>(0).get_buf();
            auto &buf_hpart
                = pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("hpart")).get_buf();

            auto &buf_field_to_render = field_getter(cur_p, pdat);

            u32 obj_cnt = main_field.get_obj_cnt();

            RTree tree(
                shamsys::instance::get_compute_queue(),
                {box.lower, box.upper},
                buf_xyz,
                obj_cnt,
                solver_config.tree_reduction_level);

            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());

            RadixTreeField<Tscal> hmax_tree = tree.compute_int_boxes(
                shamsys::instance::get_compute_queue(),
                pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("hpart")).get_buf(),
                1);

            std::vector<sycl::event> depends_list;
            Tfield *render_field = ret.get_write_access(depends_list);

            sycl::event e2 = shamsys::instance::get_compute_queue().submit([&, render_field](
                                                                               sycl::handler &cgh) {
                cgh.depends_on(depends_list);

                shamrock::tree::ObjectIterator particle_looper(tree, cgh);
                sycl::accessor xyz{shambase::get_check_ref(buf_xyz), cgh, sycl::read_only};
                sycl::accessor hpart{shambase::get_check_ref(buf_hpart), cgh, sycl::read_only};
                sycl::accessor torender{
                    shambase::get_check_ref(buf_field_to_render), cgh, sycl::read_only};

                sycl::accessor hmax{
                    shambase::get_check_ref(hmax_tree.radix_tree_field_buf), cgh, sycl::read_only};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal partmass = solver_config.gpart_mass;

                shambase::parralel_for(cgh, nx * ny, "compute slice render", [=](u32 gid) {
                    u32 ix          = gid % nx;
                    u32 iy          = gid / nx;
                    f64 fx          = ((f64(ix) + 0.5) / nx) - 0.5;
                    f64 fy          = ((f64(iy) + 0.5) / ny) - 0.5;
                    Tvec pos_render = center + delta_x * fx + delta_y * fy;

                    Tfield ret = 0;

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

            ret.complete_event_state(e2);
        });

        shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);

        return ret;
    }

} // namespace shammodels::sph::modules

using namespace shammath;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M4>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M6>;
template class shammodels::sph::modules::CartesianRender<f64_3, f64, M8>;
