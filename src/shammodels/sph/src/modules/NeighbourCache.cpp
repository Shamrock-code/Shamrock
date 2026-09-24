// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighbourCache.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/make_ndrange.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversal.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include "shamunits/Constants.hpp"

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::start_neighbors_cache() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

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
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, pcache.scanned_cnt, obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

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
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_shared_offload() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        // a depth first traversal holds at most depth + 1 entries in its stack
        u32 tree_depth = tree.get_exact_tree_depth();
        u32 stack_size = tree_depth + 1;

        shamlog_info_ln("Cache", "patch", patch_id, "tree depth =", tree_depth);

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance, stack_size](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal rint_a = hpart[id_a] * h_tolerance;

                        Tvec xyz_a = xyz[id_a];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        u32 cnt = 0;

                        particle_looper.rtree_for(
                            stack_id,
                            stack_size,
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
                                // compute only omega_a
                                Tvec dr      = xyz_a - xyz[id_b];
                                Tscal rab2   = sycl::dot(dr, dr);
                                Tscal rint_b = hpart[id_b] * h_tolerance;

                                bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                   && rab2 > rint_b * rint_b * Rker2;

                                cnt += (no_interact) ? 0 : 1;
                            });

                        neigh_cnt[id_a] = cnt;
                    });
                };
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, pcache.scanned_cnt, obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance, stack_size](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal rint_a = hpart[id_a] * h_tolerance;

                        Tvec xyz_a = xyz[id_a];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        u32 cnt = scanned_neigh_cnt[id_a];

                        particle_looper.rtree_for(
                            stack_id,
                            stack_size,
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
                                // compute only omega_a
                                Tvec dr      = xyz_a - xyz[id_b];
                                Tscal rab2   = sycl::dot(dr, dr);
                                Tscal rint_b = hpart[id_b] * h_tolerance;

                                bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                   && rab2 > rint_b * rint_b * Rker2;

                                if (!no_interact) {
                                    neigh[cnt] = id_b;
                                }
                                cnt += (no_interact) ? 0 : 1;
                            });
                    });
                };
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_2stages() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it  = tree.get_object_iterator();
        auto leaf_it = tree.get_traverser();

        u32 leaf_cnt    = tree.get_leaf_cell_count();
        u32 intnode_cnt = tree.get_internal_cell_count();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        // start by counting number of leaf neighbours

        sham::DeviceBuffer<u32> neigh_count_leaf(
            leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{tree_field_rint, leaf_it},
            sham::MultiRef{neigh_count_leaf},
            leaf_cnt,
            [intnode_cnt](
                u32 id_a,
                const Tscal *__restrict rint_tree,
                auto leaf_looper,
                u32 *__restrict neigh_cnt) {
                u32 offset_leaf = intnode_cnt;

                Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = 0;

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                        Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                        return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                               || BBAA::cella_neigh_b(
                                   leaf_a_bmin_ext,
                                   leaf_a_bmax_ext,
                                   node_aabb.lower,
                                   node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        cnt++;
                    });

                neigh_cnt[id_a] = cnt;
            });

        //{
        //    u32 offset_leaf = intnode_cnt;
        //    sycl::host_accessor neigh_cnt{neigh_count_leaf};
        //    sycl::host_accessor pos_min_cell
        //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt)};
        //    sycl::host_accessor pos_max_cell
        //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt)};
        //
        //    for (u32 i = 0; i < 1000; i++) {
        //        if(neigh_cnt[i] > 30){
        //            logger::raw_ln(i, neigh_cnt[i], pos_max_cell[i+offset_leaf] -
        //            pos_min_cell[i+offset_leaf]);
        //        }
        //    }
        //}

        tree::ObjectCache pleaf_cache
            = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

        // fill ids of leaf neighbours

        NamedStackEntry stack_loc2{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{tree_field_rint, pleaf_cache.scanned_cnt, leaf_it},
            sham::MultiRef{pleaf_cache.index_neigh_map},
            leaf_cnt,
            [intnode_cnt](
                u32 id_a,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto leaf_looper,
                u32 *__restrict neigh) {
                u32 offset_leaf = intnode_cnt;

                Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = scanned_neigh_cnt[id_a];

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                        Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                        return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                               || BBAA::cella_neigh_b(
                                   leaf_a_bmin_ext,
                                   leaf_a_bmax_ext,
                                   node_aabb.lower,
                                   node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        neigh[cnt] = leaf_b;
                        cnt++;
                    });
            });

        // search in which leaf each parts are
        sham::DeviceBuffer<u32> leaf_part_id(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, leaf_it},
            sham::MultiRef{leaf_part_id},
            obj_cnt,
            [intnode_cnt](
                u32 id_a, const Tvec *__restrict xyz, auto leaf_looper, u32 *__restrict found_id) {
                u32 offset_leaf = intnode_cnt;

                Tvec r_a = xyz[id_a];

                u32 found_id_ = i32_max; // to ensure a crash because of out of bound
                                         // access if not found

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return BBAA::is_coord_in_range_incl_max(
                            r_a, node_aabb.lower, node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        found_id_ = leaf_b - offset_leaf;
                    });

                SHAM_ASSERT(found_id_ < offset_leaf + 1);

                found_id[id_a] = found_id_;
            });

        //{
        //    sycl::host_accessor xyz{buf_xyz};
        //    sycl::host_accessor acc {leaf_part_id};
        //
        //    for(u32 i = 0; i < obj_cnt; i++){
        //        u32 leaf_id = acc[i];
        //        if(leaf_id >= leaf_cnt){
        //            logger::raw_ln("error : i=",i,"r=",xyz[i],"leaf_id=",leaf_id);
        //        }
        //    }
        //}

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, pleaf_cache, obj_it.cell_iterator, leaf_part_id},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh_cnt) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = 0;

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });
                });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc3{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                pleaf_cache,
                pcache.scanned_cnt,
                obj_it.cell_iterator,
                leaf_part_id},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = scanned_neigh_cnt[id_a];

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
                });
            });
        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_2stages_shared_offload() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it  = tree.get_object_iterator();
        auto leaf_it = tree.get_traverser();

        // a depth first traversal holds at most depth + 1 entries in its stack
        u32 tree_depth = tree.get_exact_tree_depth();
        u32 stack_size = tree_depth + 1;

        shamlog_info_ln("Cache", "patch", patch_id, "tree depth =", tree_depth);

        u32 leaf_cnt    = tree.get_leaf_cell_count();
        u32 intnode_cnt = tree.get_internal_cell_count();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        // start by counting number of leaf neighbours

        sham::DeviceBuffer<u32> neigh_count_leaf(
            leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{tree_field_rint, leaf_it},
            sham::MultiRef{neigh_count_leaf},
            leaf_cnt,
            [intnode_cnt, stack_size](
                u32 n,
                const Tscal *__restrict rint_tree,
                auto leaf_looper,
                u32 *__restrict neigh_cnt) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                        Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                        Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                        Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                        Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                        u32 cnt = 0;

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                                Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                                return BBAA::cella_neigh_b(
                                           leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                       || BBAA::cella_neigh_b(
                                           leaf_a_bmin_ext,
                                           leaf_a_bmax_ext,
                                           node_aabb.lower,
                                           node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                cnt++;
                            });

                        neigh_cnt[id_a] = cnt;
                    });
                };
            });

        tree::ObjectCache pleaf_cache
            = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

        // fill ids of leaf neighbours

        NamedStackEntry stack_loc2{"fill cache"};

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{tree_field_rint, pleaf_cache.scanned_cnt, leaf_it},
            sham::MultiRef{pleaf_cache.index_neigh_map},
            leaf_cnt,
            [intnode_cnt, stack_size](
                u32 n,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto leaf_looper,
                u32 *__restrict neigh) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                        Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                        Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                        Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                        Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                        u32 cnt = scanned_neigh_cnt[id_a];

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                                Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                                return BBAA::cella_neigh_b(
                                           leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                       || BBAA::cella_neigh_b(
                                           leaf_a_bmin_ext,
                                           leaf_a_bmax_ext,
                                           node_aabb.lower,
                                           node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                neigh[cnt] = leaf_b;
                                cnt++;
                            });
                    });
                };
            });

        // search in which leaf each parts are
        sham::DeviceBuffer<u32> leaf_part_id(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, leaf_it},
            sham::MultiRef{leaf_part_id},
            obj_cnt,
            [intnode_cnt, stack_size](
                u32 n, const Tvec *__restrict xyz, auto leaf_looper, u32 *__restrict found_id) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tvec r_a = xyz[id_a];

                        u32 found_id_ = i32_max; // to ensure a crash because of out of
                                                 // bound access if not found

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                return BBAA::is_coord_in_range_incl_max(
                                    r_a, node_aabb.lower, node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                found_id_ = leaf_b - offset_leaf;
                            });

                        SHAM_ASSERT(found_id_ < offset_leaf + 1);

                        found_id[id_a] = found_id_;
                    });
                };
            });

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, pleaf_cache, obj_it.cell_iterator, leaf_part_id},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh_cnt) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = 0;

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });
                });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc3{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                pleaf_cache,
                pcache.scanned_cnt,
                obj_it.cell_iterator,
                leaf_part_id},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = scanned_neigh_cnt[id_a];

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
                });
            });
        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

using namespace shammath;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M6>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M8>;

template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C2>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C6>;
