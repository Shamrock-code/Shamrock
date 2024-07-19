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

    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> rho_face_zm;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
        sycl::buffer<Tvec> &buf_grad_rho = shambase::get_check_ref(storage.grad_rho.get().get_buf(id));

        logger::debug_ln("Face Interpolate", "patch",id,"intepolate rho");

        rho_face_xp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        rho_face_xm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        rho_face_yp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        rho_face_ym.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        rho_face_zp.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
        rho_face_zm.add_obj(
            id,
            compute_link_field<RhoInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_rho, buf_grad_rho));
    });

    storage.rho_face_xp.set(std::move(rho_face_xp));
    storage.rho_face_xm.set(std::move(rho_face_xm));
    storage.rho_face_yp.set(std::move(rho_face_yp));
    storage.rho_face_ym.set(std::move(rho_face_ym));
    storage.rho_face_zp.set(std::move(rho_face_zp));
    storage.rho_face_zm.set(std::move(rho_face_zm));
}


template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_v_to_face() {

    class VelInterpolate{

        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;

        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_vel_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dx_v_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dy_v_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_dz_v_cell;

        VelInterpolate(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size, 
            sycl::buffer<Tvec> & vel_cell,
            sycl::buffer<Tvec> & dx_v_cell,
            sycl::buffer<Tvec> & dy_v_cell,
            sycl::buffer<Tvec> & dz_v_cell)
            : shift_get(cgh, aabb_block_lower, aabb_cell_size),
            acc_vel_cell{vel_cell, cgh, sycl::read_only},
            acc_dx_v_cell{dx_v_cell, cgh, sycl::read_only},
            acc_dy_v_cell{dy_v_cell, cgh, sycl::read_only},
            acc_dz_v_cell{dz_v_cell, cgh, sycl::read_only}
             {}


        std::array<Tvec,2> get_link_field_val(u32 id_a, u32 id_b) const {

            auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);
            
            Tvec v_a = acc_vel_cell[id_a] ;
            Tvec dx_vel_a = acc_dx_v_cell[id_a];
            Tvec dy_vel_a = acc_dy_v_cell[id_a];
            Tvec dz_vel_a = acc_dz_v_cell[id_a];

            Tvec v_b = acc_vel_cell[id_b];
            Tvec dx_vel_b = acc_dx_v_cell[id_b];
            Tvec dy_vel_b = acc_dy_v_cell[id_b];
            Tvec dz_vel_b = acc_dz_v_cell[id_b];

            Tvec vel_face_a = v_a + shift_a.x()*dx_vel_a + shift_a.y()*dy_vel_a + shift_a.z()*dz_vel_a;
            Tvec vel_face_b = v_b + shift_b.x()*dx_vel_b + shift_b.y()*dy_vel_b + shift_b.z()*dz_vel_b;

            return {vel_face_a, vel_face_b};

        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tvec,2>>> vel_face_zm;

    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tvec> &buf_vel = shambase::get_check_ref(storage.vel.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dx_vel = shambase::get_check_ref(storage.dx_v.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dy_vel = shambase::get_check_ref(storage.dy_v.get().get_buf(id));
        sycl::buffer<Tvec> &buf_dz_vel = shambase::get_check_ref(storage.dz_v.get().get_buf(id));

        logger::debug_ln("Face Interpolate", "patch",id,"intepolate vel");

        vel_face_xp.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
        vel_face_xm.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
        vel_face_yp.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
        vel_face_ym.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
        vel_face_zp.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
        vel_face_zm.add_obj(
            id,
            compute_link_field<VelInterpolate, std::array<Tvec,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes,buf_vel, buf_dx_vel, buf_dy_vel, buf_dz_vel));
    });

    storage.vel_face_xp.set(std::move(vel_face_xp));
    storage.vel_face_xm.set(std::move(vel_face_xm));
    storage.vel_face_yp.set(std::move(vel_face_yp));
    storage.vel_face_ym.set(std::move(vel_face_ym));
    storage.vel_face_zp.set(std::move(vel_face_zp));
    storage.vel_face_zm.set(std::move(vel_face_zm));

}


template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_P_to_face() {

    class PressInterpolate {
        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;


        sycl::accessor<Tscal, 1, sycl::access::mode::read, sycl::target::device> acc_P_cell;
        sycl::accessor<Tvec, 1, sycl::access::mode::read, sycl::target::device> acc_grad_P_cell;

        PressInterpolate(
            sycl::handler &cgh,
            sycl::buffer<Tvec> &aabb_block_lower,
            sycl::buffer<Tscal> &aabb_cell_size, 
            sycl::buffer<Tscal> & P_cell,
            sycl::buffer<Tvec> & grad_P_cell)
            : shift_get(cgh, aabb_block_lower, aabb_cell_size),
            acc_P_cell{P_cell, cgh, sycl::read_only},
            acc_grad_P_cell{grad_P_cell, cgh, sycl::read_only} {}


        std::array<Tscal,2> get_link_field_val(u32 id_a, u32 id_b) const {

            auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);
            
            Tscal P_a = acc_P_cell[id_a];
            Tvec grad_P_a = acc_grad_P_cell[id_a];
            Tscal P_b = acc_P_cell[id_b];
            Tvec grad_P_b = acc_grad_P_cell[id_b];

            Tscal P_face_a = P_a + sycl::dot(grad_P_a, shift_a);
            Tscal P_face_b = P_b + sycl::dot(grad_P_b, shift_b);

            return {P_face_a, P_face_b};

        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_xp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_xm;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_yp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_ym;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_zp;
    shambase::DistributedData<NeighGraphLinkField<std::array<Tscal,2>>> press_face_zm;


    storage.cell_link_graph.get().for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat                     = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        sycl::buffer<Tscal> &buf_press = shambase::get_check_ref(storage.press.get().get_buf(id));
        sycl::buffer<Tvec> &buf_grad_P = shambase::get_check_ref(storage.grad_P.get().get_buf(id));

        logger::debug_ln("Face Interpolate", "patch",id,"intepolate press");

        press_face_xp.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
        press_face_xm.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
        press_face_yp.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
        press_face_ym.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
        press_face_zp.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
        press_face_zm.add_obj(
            id,
            compute_link_field<PressInterpolate, std::array<Tscal,2>>(
                q,
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                cell0block_aabb_lower,
                block_cell_sizes, buf_press, buf_grad_P));
    });

    storage.press_face_xp.set(std::move(press_face_xp));
    storage.press_face_xm.set(std::move(press_face_xm));
    storage.press_face_yp.set(std::move(press_face_yp));
    storage.press_face_ym.set(std::move(press_face_ym));
    storage.press_face_zp.set(std::move(press_face_zp));
    storage.press_face_zm.set(std::move(press_face_zm));
}


template class shammodels::basegodunov::modules::FaceInterpolate<f64_3, i64_3>;