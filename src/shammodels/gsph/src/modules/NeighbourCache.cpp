// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighbourCache.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief GSPH neighbor cache builder implementation
 */

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/gsph/modules/NeighbourCache.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include "shamtree/kernels/geometry_utils.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    shamrock::tree::ObjectCache NeighbourCache<Tvec, SPHKernel>::build_patch_cache(
        u64 patch_id, Tscal h_tolerance) {

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        auto &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamsys::instance::get_compute_queue().wait_and_throw();

        // First pass: count neighbors
        {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            auto xyz             = buf_xyz.get_read_access(depends_list);
            auto hpart           = buf_hpart.get_read_access(depends_list);
            auto rint_tree       = tree_field_rint.get_read_access(depends_list);
            auto neigh_cnt       = neigh_count.get_write_access(depends_list);
            auto particle_looper = obj_it.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&, h_tolerance](sycl::handler &cgh) {
                shambase::parallel_for(cgh, obj_cnt, "gsph_count_neighbors", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a   = xyz[id_a];

                    Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                    Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                    u32 cnt = 0;

                    particle_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact
                                = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            cnt += (no_interact) ? 0 : 1;
                        });

                    neigh_cnt[id_a] = cnt;
                });
            });

            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            neigh_count.complete_event_state(e);
            tree_field_rint.complete_event_state(e);
            obj_it.complete_event_state(e);
        }

        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        // Second pass: fill neighbor indices
        {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            sham::EventList depends_list;

            auto xyz               = buf_xyz.get_read_access(depends_list);
            auto hpart             = buf_hpart.get_read_access(depends_list);
            auto rint_tree         = tree_field_rint.get_read_access(depends_list);
            auto scanned_neigh_cnt = pcache.scanned_cnt.get_read_access(depends_list);
            auto neigh             = pcache.index_neigh_map.get_write_access(depends_list);
            auto particle_looper   = obj_it.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&, h_tolerance](sycl::handler &cgh) {
                shambase::parallel_for(cgh, obj_cnt, "gsph_fill_neighbors", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    Tscal rint_a = hpart[id_a] * h_tolerance;
                    Tvec xyz_a   = xyz[id_a];

                    Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                    Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                    u32 write_idx = scanned_neigh_cnt[id_a];

                    particle_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        },
                        [&](u32 id_b) {
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact
                                = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            if (!no_interact) {
                                neigh[write_idx++] = id_b;
                            }
                        });
                });
            });

            buf_xyz.complete_event_state(e);
            buf_hpart.complete_event_state(e);
            tree_field_rint.complete_event_state(e);
            pcache.scanned_cnt.complete_event_state(e);
            pcache.index_neigh_map.complete_event_state(e);
            obj_it.complete_event_state(e);
        }

        return pcache;
    }

    template<class Tvec, template<class> class SPHKernel>
    void NeighbourCache<Tvec, SPHKernel>::build_cache() {
        StackEntry stack_loc{};

        shambase::Timer time_neigh;
        time_neigh.start();

        // c_smooth defaults to 1.0 for Newtonian, larger for SR
        Tscal h_tolerance = solver_config.htol_up_coarse_cycle * solver_config.c_smooth;

        shambase::get_check_ref(storage.neigh_cache).free_alloc();

        using namespace shamrock::patch;
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            auto &ncache = shambase::get_check_ref(storage.neigh_cache);
            ncache.neigh_cache.add_obj(
                cur_p.id_patch, build_patch_cache(cur_p.id_patch, h_tolerance));
        });

        time_neigh.end();
        storage.timings_details.neighbors += time_neigh.elasped_sec();
    }

    // Explicit instantiations
    template class NeighbourCache<f64_3, shammath::M4>;
    template class NeighbourCache<f64_3, shammath::M6>;
    template class NeighbourCache<f64_3, shammath::M8>;
    template class NeighbourCache<f64_3, shammath::C2>;
    template class NeighbourCache<f64_3, shammath::C4>;
    template class NeighbourCache<f64_3, shammath::C6>;
    template class NeighbourCache<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules
