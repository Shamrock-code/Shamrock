// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRTree.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/zeus/modules/AMRTree.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::AMRTree<Tvec, TgridVec>::build_trees() {

    StackEntry stack_loc{};

    using MergedPDat = shambase::DistributedData<shamrock::MergedPatchData>;
    using RTree      = typename Storage::RTree;

    MergedPDat &mpdat = storage.merged_patchdata_ghost.get();

    shamrock::patch::PatchDataLayout &mpdl = storage.ghost_layout.get();

    u32 reduc_level = 0;

    storage.merge_patch_bounds.set(
        mpdat.map<shammath::AABB<TgridVec>>([&](u64 id, shamrock::MergedPatchData &merged) {
            logger::debug_ln("AMR", "compute bound merged patch", id);

            TgridVec min_bound = merged.pdat.get_field<TgridVec>(0).compute_min();
            TgridVec max_bound = merged.pdat.get_field<TgridVec>(1).compute_max();

            return shammath::AABB<TgridVec>{min_bound, max_bound};
        }));

    shambase::DistributedData<shammath::AABB<TgridVec>> &bounds = storage.merge_patch_bounds.get();

    shambase::DistributedData<RTree> trees
        = mpdat.map<RTree>([&](u64 id, shamrock::MergedPatchData &merged) {
              logger::debug_ln("AMR", "compute tree for merged patch", id);

              auto aabb = bounds.get(id);

              TgridVec bmin = aabb.lower;
              TgridVec bmax = aabb.upper;

              TgridVec diff = bmax - bmin;
              diff.x()      = shambase::roundup_pow2(diff.x());
              diff.y()      = shambase::roundup_pow2(diff.y());
              diff.z()      = shambase::roundup_pow2(diff.z());
              bmax          = bmin + diff;

              auto &field_pos = merged.pdat.get_field<TgridVec>(0);

              RTree tree(
                  shamsys::instance::get_compute_queue(),
                  {bmin, bmax},
                  field_pos.get_buf(),
                  field_pos.get_obj_cnt(),
                  reduc_level);

              return tree;
          });

    trees.for_each([](u64 id, RTree &tree) {
        tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
        tree.convert_bounding_box(shamsys::instance::get_compute_queue());
    });

    storage.trees.set(std::move(trees));
}

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::AMRTree<Tvec, TgridVec>::correct_bounding_box() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    storage.trees.get().for_each([&](u64 id, RTree &tree) {
        u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count           = leaf_count + internal_cell_count;

        sycl::buffer<TgridVec> tmp_min_cell(tot_count);
        sycl::buffer<TgridVec> tmp_max_cell(tot_count);

        MergedPDat &mpdat                    = storage.merged_patchdata_ghost.get().get(id);
        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator cell_looper(tree, cgh);

            u32 leaf_offset = tree.tree_struct.internal_cell_count;

            sycl::accessor acc_bmin{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor acc_bmax{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor comp_min{tmp_min_cell, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor comp_max{tmp_max_cell, cgh, sycl::write_only, sycl::no_init};

            TgridVec imin = shambase::VectorProperties<TgridVec>::get_max();
            TgridVec imax = shambase::VectorProperties<TgridVec>::get_min();

            shambase::parralel_for(cgh, leaf_count, "compute leaf boxes", [=](u64 leaf_id) {
                TgridVec min = imin;
                TgridVec max = imax;

                cell_looper.iter_object_in_cell(leaf_id + leaf_offset, [&](u32 block_id) {
                    TgridVec bmin = acc_bmin[block_id];
                    TgridVec bmax = acc_bmax[block_id];

                    min = sham::min(min, bmin);
                    max = sham::max(max, bmax);
                });

                comp_min[leaf_offset + leaf_id] = min;
                comp_max[leaf_offset + leaf_id] = max;
            });
        });

        auto ker_reduc_hmax = [&](sycl::handler &cgh) {
            u32 offset_leaf = internal_cell_count;

            sycl::accessor comp_min{tmp_min_cell, cgh, sycl::read_write};
            sycl::accessor comp_max{tmp_max_cell, cgh, sycl::read_write};

            sycl::accessor rchild_id{
                shambase::get_check_ref(tree.tree_struct.buf_rchild_id), cgh, sycl::read_only};
            sycl::accessor lchild_id{
                shambase::get_check_ref(tree.tree_struct.buf_lchild_id), cgh, sycl::read_only};
            sycl::accessor rchild_flag{
                shambase::get_check_ref(tree.tree_struct.buf_rchild_flag), cgh, sycl::read_only};
            sycl::accessor lchild_flag{
                shambase::get_check_ref(tree.tree_struct.buf_lchild_flag), cgh, sycl::read_only};

            shambase::parralel_for(cgh, internal_cell_count, "propagate up", [=](u64 gid) {
                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                TgridVec bminl = comp_min[lid];
                TgridVec bminr = comp_min[rid];
                TgridVec bmaxl = comp_max[lid];
                TgridVec bmaxr = comp_max[rid];

                TgridVec bmin = sham::min(bminl, bminr);
                TgridVec bmax = sham::max(bmaxl, bmaxr);

                comp_min[gid] = bmin;
                comp_max[gid] = bmax;
            });
        };

        for (u32 i = 0; i < tree.tree_depth; i++) {
            shamsys::instance::get_compute_queue().submit(ker_reduc_hmax);
        }

        sycl::buffer<TgridVec> &tree_bmin
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> &tree_bmax
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator cell_looper(tree, cgh);

            u32 leaf_offset = tree.tree_struct.internal_cell_count;

            sycl::accessor comp_bmin{tmp_min_cell, cgh, sycl::read_only};
            sycl::accessor comp_bmax{tmp_max_cell, cgh, sycl::read_only};

            sycl::accessor tree_buf_min{tree_bmin, cgh, sycl::read_write};
            sycl::accessor tree_buf_max{tree_bmax, cgh, sycl::read_write};

            shambase::parralel_for(cgh, tot_count, "write in tree range", [=](u64 nid) {
                TgridVec load_min = comp_bmin[nid];
                TgridVec load_max = comp_bmax[nid];

                // if(
                //     (!shambase::vec_equals(load_min,tree_buf_min[nid]))
                //   || !shambase::vec_equals(load_max,tree_buf_max[nid])
                //     )
                //{
                //
                //    sycl::ext::oneapi::experimental::printf(
                //        "%ld : (%ld %ld %ld) -> (%ld %ld %ld) & (%ld %ld %ld) -> (%ld %ld %ld)\n",
                //        nid,
                //        tree_buf_min[nid].x(),tree_buf_min[nid].y(),tree_buf_min[nid].z(),
                //        load_min.x(),load_min.y(),load_min.z(),
                //        tree_buf_max[nid].x(),tree_buf_max[nid].y(),tree_buf_max[nid].z(),
                //        load_max.x(),load_max.y(),load_max.z()
                //    );
                //
                //}

                tree_buf_min[nid] = load_min;
                tree_buf_max[nid] = load_max;
            });
        });
    });
}

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::AMRTree<Tvec, TgridVec>::build_neigh_cache() {

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    storage.neighbors_cache.set(shamrock::tree::ObjectCacheHandler(u64(10e9), [&](u64 patch_id) {
        logger::debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        MergedPDat &mfield = storage.merged_patchdata_ghost.get().get(patch_id);

        sycl::buffer<TgridVec> &buf_cell_min = mfield.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mfield.pdat.get_field_buf_ref<TgridVec>(1);

        RTree &tree = storage.trees.get().get(patch_id);

        u32 obj_cnt = mfield.total_elements;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sycl::buffer<u32> neigh_count(obj_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::ObjectIterator cell_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor neigh_cnt{neigh_count, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32) gid;

                shammath::AABB<TgridVec> cell_aabb{cell_min[id_a], cell_max[id_a]};

                u32 cnt = 0;

                cell_looper.rtree_for(
                    [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                        return shammath::AABB<TgridVec>{bmin, bmax}
                            .get_intersect(cell_aabb)
                            .is_surface_or_volume();
                    },
                    [&](u32 id_b) {
                        bool no_interact = !shammath::AABB<TgridVec>{cell_min[id_b], cell_max[id_b]}
                                                .get_intersect(cell_aabb)
                                                .is_surface();

                        cnt += (no_interact) ? 0 : 1;
                    });

                // if(cell_aabb.lower.x() < 0 && cell_aabb.lower.y() == 10){
                //     sycl::ext::oneapi::experimental::printf("%d (%ld %ld %ld) : %d\n", id_a,
                //     cell_aabb.lower.x(),cell_aabb.lower.y(),cell_aabb.lower.z()
                //     ,cnt);
                // }

                neigh_cnt[id_a] = cnt;
            });
        });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::ObjectIterator cell_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor scanned_neigh_cnt{pcache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pcache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32) gid;

                shammath::AABB<TgridVec> cell_aabb{cell_min[id_a], cell_max[id_a]};

                u32 cnt = scanned_neigh_cnt[id_a];

                cell_looper.rtree_for(
                    [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                        return shammath::AABB<TgridVec>{bmin, bmax}
                            .get_intersect(cell_aabb)
                            .is_surface_or_volume();
                    },
                    [&](u32 id_b) {
                        bool no_interact = !shammath::AABB<TgridVec>{cell_min[id_b], cell_max[id_b]}
                                                .get_intersect(cell_aabb)
                                                .is_surface();

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });
        });

        return pcache;
    }));

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        storage.neighbors_cache.get().preload(cur_p.id_patch);
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();
}

template class shammodels::zeus::modules::AMRTree<f64_3, i64_3>;
