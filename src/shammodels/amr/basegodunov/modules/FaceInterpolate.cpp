// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FaceInterpolate.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/FaceInterpolate.hpp"

#include "shammodels/amr/NeighGraphLinkField.hpp"

namespace {

    template<class Tvec, class TgridVec, class AMRBlock>
    class GetShift {

        public:
        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device>
            acc_aabb_block_lower;
        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device> acc_aabb_cell_size;

        GetShift(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size)
            : acc_aabb_block_lower{aabb_block_lower, cgh, sycl::read_only},
              acc_aabb_cell_size{aabb_cell_size, cgh, sycl::read_only} {}

        shammath::AABB<Tvec> get_cell_aabb(u32 id) const {

            const u32 cell_global_id = (u32) id;

            const u32 block_id    = cell_global_id / AMRBlock::block_size;
            const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

            // fetch current block info
            const Tvec cblock_min  = acc_aabb_block_lower[block_id];
            const Tscal delta_cell = acc_aabb_cell_size[block_id];

            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
            Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

            Tvec aabb_min = cblock_min + offset;
            Tvec aabb_max = aabb_min + delta_cell;

            return {aabb_min, aabb_max};
        }

        std::pair<Tvec, Tvec> get_shifts(u32 id_a, u32 id_b) const {

            shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
            shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

            shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

            Tvec face_center = face_aabb.get_center();

            Tvec shift_a = face_center - aabb_cell_a.get_center();
            Tvec shift_b = face_center - aabb_cell_b.get_center();

            return {shift_a, shift_b};
        }
    };

} // namespace

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_rho_to_face() {

    class RhoInterpolate {
        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;


        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device> acc_rho_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_grad_rho_cell;

        RhoInterpolate(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size, 
            sycl::buffer<Tscal> & rho_cell,
            sycl::buffer<Tvec> & grad_rho_cell)
            : shift_get(cgh, aabb_block_lower, aabb_cell_size),
            acc_rho_cell{rho_cell, cgh, sycl::read_only},
            acc_grad_rho_cell{grad_rho_cell, cgh, sycl::read_only} {}


        std::array<Tscal,2> get_link_field_val(u32 id_a, u32 id_b) const {

            auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);
            
            Tscal rho_a = acc_rho_cell[id_a];
            Tvec grad_rho_a = acc_grad_rho_cell[id_a];
            Tscal rho_b = acc_rho_cell[id_b];
            Tvec grad_rho_b = acc_grad_rho_cell[id_b];

            Tscal rho_face_a = rho_a + sycl::dot(grad_rho_a, shift_a);
            Tscal rho_face_b = rho_b + sycl::dot(grad_rho_b, shift_b);

            return {rho_face_a, rho_face_b};

        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    u32 ndust;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_zm;


    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_dust_face_zm;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irho_dust_ghost                            = ghost_layout.get_field_idx<Tscal>("rho_dust", ndust);

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
        sycl::buffer<Tvec> &buf_grad_rho = shambase::get_check_ref(storage.grad_rho.get().get_buf(id));

        sycl::buffer<Tscal> &buf_rho_dust = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);
        sycl::buffer<Tvec> &buf_grad_rho_dust = shambase::get_check_ref(storage.grad_rho_dust.get().get_buf(id));

        logger::debug_ln("Face Interpolate", "patch",id,"intepolate rho");

        rho_face_xp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));

        rho_dust_face_xp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));


        rho_face_xm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        
        rho_dust_face_xm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));


        rho_face_yp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        
        rho_dust_face_yp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));


        rho_face_ym.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));

        rho_dust_face_ym.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));


        rho_face_zp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));

        rho_dust_face_zp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));


        rho_face_zm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        
        rho_dust_face_zm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho_dust, buf_grad_rho_dust));
    });

    storage.rho_face_xp.set(std::move(rho_face_xp));
    storage.rho_face_xm.set(std::move(rho_face_xm));
    storage.rho_face_yp.set(std::move(rho_face_yp));
    storage.rho_face_ym.set(std::move(rho_face_ym));
    storage.rho_face_zp.set(std::move(rho_face_zp));
    storage.rho_face_zm.set(std::move(rho_face_zm));

    storage.rho_dust_face_xp.set(std::move(rho_dust_face_xp));
    storage.rho_dust_face_xm.set(std::move(rho_dust_face_xm));
    storage.rho_dust_face_yp.set(std::move(rho_dust_face_yp));
    storage.rho_dust_face_ym.set(std::move(rho_dust_face_ym));
    storage.rho_dust_face_zp.set(std::move(rho_dust_face_zp));
    storage.rho_dust_face_zm.set(std::move(rho_dust_face_zm));

}


template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_rhov_to_face() {

    class RhovInterpolate{

        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;


        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_rhov_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dx_rhov_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dy_rhov_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dz_rhov_cell;

        RhovInterpolate(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size, 
            sycl::buffer<Tvec> & rhov_cell,
            sycl::buffer<Tvec> & dx_rhov_cell,
            sycl::buffer<Tvec> & dy_rhov_cell,
            sycl::buffer<Tvec> & dz_rhov_cell)
            : shift_get(cgh, aabb_block_lower, aabb_cell_size),
            acc_rhov_cell{rhov_cell, cgh, sycl::read_only},
            acc_dx_rhov_cell{dx_rhov_cell, cgh, sycl::read_only},
            acc_dy_rhov_cell{dy_rhov_cell, cgh, sycl::read_only},
            acc_dz_rhov_cell{dz_rhov_cell, cgh, sycl::read_only}
             {}


        std::array<Tvec,2> get_link_field_val(u32 id_a, u32 id_b) const {

            auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);
            
            Tvec rhov_a = acc_rhov_cell[id_a];
            Tvec dx_rhov_a = acc_dx_rhov_cell[id_a];
            Tvec dy_rhov_a = acc_dy_rhov_cell[id_a];
            Tvec dz_rhov_a = acc_dz_rhov_cell[id_a];

            Tvec rhov_b = acc_rhov_cell[id_b];
            Tvec dx_rhov_b = acc_dx_rhov_cell[id_b];
            Tvec dy_rhov_b = acc_dy_rhov_cell[id_b];
            Tvec dz_rhov_b = acc_dz_rhov_cell[id_b];

            Tvec rhov_face_a = rhov_a + shift_a.x()*dx_rhov_a + shift_a.y()*dy_rhov_a + shift_a.z()*dz_rhov_a;
            Tvec rhov_face_b = rhov_b + shift_b.x()*dx_rhov_b + shift_b.y()*dy_rhov_b + shift_b.z()*dz_rhov_b;

            return {rhov_face_a, rhov_face_b};

        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    u32 ndust;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_face_zm;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> rhov_dust_face_zm;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irhov_ghost                                 = ghost_layout.get_field_idx<Tvec>("rhovel");

    u32 irhov_dust_ghost                                 = ghost_layout.get_field_idx<Tvec>("rhovel_dust", ndust );


    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tvec> &buf_rhov = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_ghost);
        sycl::buffer<Tvec> &buf_dx_rhov = shambase::get_check_ref(storage.dx_rhov.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dy_rhov = shambase::get_check_ref(storage.dy_rhov.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dz_rhov = shambase::get_check_ref(storage.dz_rhov.get().get_buf(id));

        sycl::buffer<Tvec> &buf_rhov_dust = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_dust_ghost);
        sycl::buffer<Tvec> &buf_dx_rhov_dust = shambase::get_check_ref(storage.dx_rhov_dust.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dy_rhov_dust = shambase::get_check_ref(storage.dy_rhov_dust.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dz_rhov_dust = shambase::get_check_ref(storage.dz_rhov_dust.get().get_buf(id));


        logger::debug_ln("Face Interpolate", "patch",id,"intepolate rhov");

        rhov_face_xp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_xp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));


        rhov_face_xm.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_xm.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));


        rhov_face_yp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_yp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));


        rhov_face_ym.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_ym.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));

        rhov_face_zp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_zp.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));


        rhov_face_zm.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov, buf_dx_rhov, buf_dy_rhov, buf_dz_rhov));
        
        rhov_dust_face_zm.add_obj(
            id,
            compute_link_field<RhovInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_rhov_dust, buf_dx_rhov_dust, buf_dy_rhov_dust, buf_dz_rhov_dust));
    });

    storage.rhov_face_xp.set(std::move(rhov_face_xp));
    storage.rhov_face_xm.set(std::move(rhov_face_xm));
    storage.rhov_face_yp.set(std::move(rhov_face_yp));
    storage.rhov_face_ym.set(std::move(rhov_face_ym));
    storage.rhov_face_zp.set(std::move(rhov_face_zp));
    storage.rhov_face_zm.set(std::move(rhov_face_zm));

    storage.rhov_dust_face_xp.set(std::move(rhov_dust_face_xp));
    storage.rhov_dust_face_xm.set(std::move(rhov_dust_face_xm));
    storage.rhov_dust_face_yp.set(std::move(rhov_dust_face_yp));
    storage.rhov_dust_face_ym.set(std::move(rhov_dust_face_ym));
    storage.rhov_dust_face_zp.set(std::move(rhov_dust_face_zp));
    storage.rhov_dust_face_zm.set(std::move(rhov_dust_face_zm));

}


template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_rhoe_to_face() {

    class RhoeInterpolate {
        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;


        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device> acc_rhoe_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_grad_rhoe_cell;

        RhoeInterpolate(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size, 
            sycl::buffer<Tscal> & rhoe_cell,
            sycl::buffer<Tvec> & grad_rhoe_cell)
            : shift_get(cgh, aabb_block_lower, aabb_cell_size),
            acc_rhoe_cell{rhoe_cell, cgh, sycl::read_only},
            acc_grad_rhoe_cell{grad_rhoe_cell, cgh, sycl::read_only} {}


        std::array<Tscal,2> get_link_field_val(u32 id_a, u32 id_b) const {

            auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);
            
            Tscal rhoe_a = acc_rhoe_cell[id_a];
            Tvec grad_rhoe_a = acc_grad_rhoe_cell[id_a];
            Tscal rhoe_b = acc_rhoe_cell[id_b];
            Tvec grad_rhoe_b = acc_grad_rhoe_cell[id_b];

            Tscal rhoe_face_a = rhoe_a + sycl::dot(grad_rhoe_a, shift_a);
            Tscal rhoe_face_b = rhoe_b + sycl::dot(grad_rhoe_b, shift_b);

            return {rhoe_face_a, rhoe_face_b};

        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rhoe_face_zm;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irhoe_ghost                                 = ghost_layout.get_field_idx<Tscal>("rhoetot");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tscal> &buf_rhoe = mpdat.pdat.get_field_buf_ref<Tscal>(irhoe_ghost);
        sycl::buffer<Tvec> &buf_grad_rhoe = shambase::get_check_ref(storage.grad_rho.get().get_buf(id));

        logger::debug_ln("Face Interpolate", "patch",id,"intepolate rhoe");

        rhoe_face_xp.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
        rhoe_face_xm.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
        rhoe_face_yp.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
        rhoe_face_ym.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
        rhoe_face_zp.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
        rhoe_face_zm.add_obj(
            id,
            compute_link_field<RhoeInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rhoe, buf_grad_rhoe));
    });

    storage.rhoe_face_xp.set(std::move(rhoe_face_xp));
    storage.rhoe_face_xm.set(std::move(rhoe_face_xm));
    storage.rhoe_face_yp.set(std::move(rhoe_face_yp));
    storage.rhoe_face_ym.set(std::move(rhoe_face_ym));
    storage.rhoe_face_zp.set(std::move(rhoe_face_zp));
    storage.rhoe_face_zm.set(std::move(rhoe_face_zm));
}


template class shammodels::basegodunov::modules::FaceInterpolate<f64_3, i64_3>;