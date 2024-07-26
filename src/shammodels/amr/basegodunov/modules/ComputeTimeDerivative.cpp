// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeTimeDerivative.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/ComputeTimeDerivative.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class T>
using NGLink = shammodels::basegodunov::modules::NeighGraphLinkField<T>;

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeTimeDerivative<Tvec, TgridVec>::compute_dt_fields() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> cfield_dtrho = utility.make_compute_field<Tscal>("dt rho", AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_dtrhov = utility.make_compute_field<Tvec>("dt rhovel", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_dtrhoe = utility.make_compute_field<Tscal>("dt rhoe", AMRBlock::block_size);

    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_xp  = storage.flux_rho_face_xp .get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_xm  = storage.flux_rho_face_xm .get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_yp  = storage.flux_rho_face_yp .get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_ym  = storage.flux_rho_face_ym .get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_zp  = storage.flux_rho_face_zp .get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rho_face_zm  = storage.flux_rho_face_zm .get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_xp = storage.flux_rhov_face_xp.get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_xm = storage.flux_rhov_face_xm.get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_yp = storage.flux_rhov_face_yp.get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_ym = storage.flux_rhov_face_ym.get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_zp = storage.flux_rhov_face_zp.get();
    shambase::DistributedData<NGLink<Tvec>>  & flux_rhov_face_zm = storage.flux_rhov_face_zm.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_xp = storage.flux_rhoe_face_xp.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_xm = storage.flux_rhoe_face_xm.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_yp = storage.flux_rhoe_face_yp.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_ym = storage.flux_rhoe_face_ym.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_zp = storage.flux_rhoe_face_zp.get();
    shambase::DistributedData<NGLink<Tscal>> & flux_rhoe_face_zm = storage.flux_rhoe_face_zm.get();

    scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p, shamrock::patch::PatchData & pdat){

        logger::debug_ln("[AMR Flux]", "accumulate fluxes patch",p.id_patch);

        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        u32 id = p.id_patch;
        OrientedAMRGraph &oriented_cell_graph = storage.cell_link_graph.get().get(id);
        
        NGLink<Tscal> & patch_flux_rho_face_xp  = flux_rho_face_xp .get(id);
        NGLink<Tscal> & patch_flux_rho_face_xm  = flux_rho_face_xm .get(id);
        NGLink<Tscal> & patch_flux_rho_face_yp  = flux_rho_face_yp .get(id);
        NGLink<Tscal> & patch_flux_rho_face_ym  = flux_rho_face_ym .get(id);
        NGLink<Tscal> & patch_flux_rho_face_zp  = flux_rho_face_zp .get(id);
        NGLink<Tscal> & patch_flux_rho_face_zm  = flux_rho_face_zm .get(id);

        NGLink<Tvec>  & patch_flux_rhov_face_xp = flux_rhov_face_xp.get(id);
        NGLink<Tvec>  & patch_flux_rhov_face_xm = flux_rhov_face_xm.get(id);
        NGLink<Tvec>  & patch_flux_rhov_face_yp = flux_rhov_face_yp.get(id);
        NGLink<Tvec>  & patch_flux_rhov_face_ym = flux_rhov_face_ym.get(id);
        NGLink<Tvec>  & patch_flux_rhov_face_zp = flux_rhov_face_zp.get(id);
        NGLink<Tvec>  & patch_flux_rhov_face_zm = flux_rhov_face_zm.get(id);

        NGLink<Tscal> & patch_flux_rhoe_face_xp = flux_rhoe_face_xp.get(id);
        NGLink<Tscal> & patch_flux_rhoe_face_xm = flux_rhoe_face_xm.get(id);
        NGLink<Tscal> & patch_flux_rhoe_face_yp = flux_rhoe_face_yp.get(id);
        NGLink<Tscal> & patch_flux_rhoe_face_ym = flux_rhoe_face_ym.get(id);
        NGLink<Tscal> & patch_flux_rhoe_face_zp = flux_rhoe_face_zp.get(id);
        NGLink<Tscal> & patch_flux_rhoe_face_zm = flux_rhoe_face_zm.get(id);

        sycl::buffer<Tscal> & buf_flux_rho_face_xp  = patch_flux_rho_face_xp .link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rho_face_xm  = patch_flux_rho_face_xm .link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rho_face_yp  = patch_flux_rho_face_yp .link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rho_face_ym  = patch_flux_rho_face_ym .link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rho_face_zp  = patch_flux_rho_face_zp .link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rho_face_zm  = patch_flux_rho_face_zm .link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_xp = patch_flux_rhov_face_xp.link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_xm = patch_flux_rhov_face_xm.link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_yp = patch_flux_rhov_face_yp.link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_ym = patch_flux_rhov_face_ym.link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_zp = patch_flux_rhov_face_zp.link_graph_field;
        sycl::buffer<Tvec>  & buf_flux_rhov_face_zm = patch_flux_rhov_face_zm.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_xp = patch_flux_rhoe_face_xp.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_xm = patch_flux_rhoe_face_xm.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_yp = patch_flux_rhoe_face_yp.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_ym = patch_flux_rhoe_face_ym.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_zp = patch_flux_rhoe_face_zp.link_graph_field;
        sycl::buffer<Tscal> & buf_flux_rhoe_face_zm = patch_flux_rhoe_face_zm.link_graph_field;

        sycl::buffer<Tscal> & dt_rho_patch = cfield_dtrho.get_buf_check(id);
        sycl::buffer<Tvec> & dt_rhov_patch = cfield_dtrhov.get_buf_check(id);
        sycl::buffer<Tscal> & dt_rhoe_patch = cfield_dtrhoe.get_buf_check(id);

        AMRGraph &graph_neigh_xp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]);
        AMRGraph &graph_neigh_xm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]);
        AMRGraph &graph_neigh_yp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]);
        AMRGraph &graph_neigh_ym = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]);
        AMRGraph &graph_neigh_zp = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]);
        AMRGraph &graph_neigh_zm = shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]);

        sycl::buffer<Tscal> & block_cell_sizes = storage.cell_infos.get().block_cell_sizes.get_buf_check(id);
        sycl::buffer<Tvec> & cell0block_aabb_lower = storage.cell_infos.get().cell0block_aabb_lower.get_buf_check(id);

        u32 cell_count = pdat.get_obj_cnt()*AMRBlock::block_size;

        q.submit([&](sycl::handler & cgh){
            AMRGraphLinkiterator graph_iter_xp{graph_neigh_xp, cgh};
            AMRGraphLinkiterator graph_iter_xm{graph_neigh_xm, cgh};
            AMRGraphLinkiterator graph_iter_yp{graph_neigh_yp, cgh};
            AMRGraphLinkiterator graph_iter_ym{graph_neigh_ym, cgh};
            AMRGraphLinkiterator graph_iter_zp{graph_neigh_zp, cgh};
            AMRGraphLinkiterator graph_iter_zm{graph_neigh_zm, cgh};

            sycl::accessor acc_aabb_block_lower{cell0block_aabb_lower, cgh, sycl::read_only};
            sycl::accessor acc_aabb_cell_size{block_cell_sizes, cgh, sycl::read_only};

            sycl::accessor flux_rho_face_xp  {buf_flux_rho_face_xp , cgh, sycl::read_only};
            sycl::accessor flux_rho_face_xm  {buf_flux_rho_face_xm , cgh, sycl::read_only};
            sycl::accessor flux_rho_face_yp  {buf_flux_rho_face_yp , cgh, sycl::read_only};
            sycl::accessor flux_rho_face_ym  {buf_flux_rho_face_ym , cgh, sycl::read_only};
            sycl::accessor flux_rho_face_zp  {buf_flux_rho_face_zp , cgh, sycl::read_only};
            sycl::accessor flux_rho_face_zm  {buf_flux_rho_face_zm , cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_xp {buf_flux_rhov_face_xp, cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_xm {buf_flux_rhov_face_xm, cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_yp {buf_flux_rhov_face_yp, cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_ym {buf_flux_rhov_face_ym, cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_zp {buf_flux_rhov_face_zp, cgh, sycl::read_only};
            sycl::accessor flux_rhov_face_zm {buf_flux_rhov_face_zm, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_xp {buf_flux_rhoe_face_xp, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_xm {buf_flux_rhoe_face_xm, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_yp {buf_flux_rhoe_face_yp, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_ym {buf_flux_rhoe_face_ym, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_zp {buf_flux_rhoe_face_zp, cgh, sycl::read_only};
            sycl::accessor flux_rhoe_face_zm {buf_flux_rhoe_face_zm, cgh, sycl::read_only};

            sycl::accessor acc_dt_rho_patch  {dt_rho_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_dt_rhov_patch {dt_rhov_patch, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_dt_rhoe_patch {dt_rhoe_patch, cgh, sycl::write_only, sycl::no_init};

            auto get_cell_aabb = [=](u32 id) -> shammath::AABB<Tvec>  {

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
            };

            auto get_face_surface = [=](u32 id_a, u32 id_b) -> Tscal {

                shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
                shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

                shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

                Tvec delta_face = face_aabb.delt();

                delta_face.x() = (delta_face.x() == 0) ? 1 : delta_face.x();
                delta_face.y() = (delta_face.y() == 0) ? 1 : delta_face.y();
                delta_face.z() = (delta_face.z() == 0) ? 1 : delta_face.z();

                return delta_face.x() * delta_face.y() * delta_face.z();
            };

            shambase::parralel_for(cgh, cell_count, "accumulate fluxes", [=](u32 id_a) {

                const u32 cell_global_id = (u32) id_a;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                Tscal V_i = acc_aabb_cell_size[block_id];
                V_i = V_i*V_i*V_i;

                Tscal dtrho = 0;
                Tvec dtrhov = {0,0,0};
                Tscal dtrhoe = 0;
                
                graph_iter_xp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_xp[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_xp[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_xp[link_id]*S_ij;
                });
                
                graph_iter_yp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_yp[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_yp[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_yp[link_id]*S_ij;
                });
                
                graph_iter_zp.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_zp[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_zp[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_zp[link_id]*S_ij;
                });
                
                graph_iter_xm.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_xm[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_xm[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_xm[link_id]*S_ij;
                });
                
                graph_iter_ym.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_ym[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_ym[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_ym[link_id]*S_ij;
                });
                
                graph_iter_zm.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id){
                    Tscal S_ij = get_face_surface(id_a, id_b);
                    dtrho -= flux_rho_face_zm[link_id]*S_ij;
                    dtrhov -= flux_rhov_face_zm[link_id]*S_ij;
                    dtrhoe -= flux_rhoe_face_zm[link_id]*S_ij;
                });

                dtrho /= V_i;
                dtrhov /= V_i;
                dtrhoe /= V_i;

                acc_dt_rho_patch[id_a] = dtrho;
                acc_dt_rhov_patch[id_a] = dtrhov;
                acc_dt_rhoe_patch[id_a] = dtrhoe;

            });

        });

    });

    storage.dtrho.set(std::move(cfield_dtrho));
    storage.dtrhov.set(std::move(cfield_dtrhov));
    storage.dtrhoe.set(std::move(cfield_dtrhoe));

}

template class shammodels::basegodunov::modules::ComputeTimeDerivative<f64_3, i64_3>;