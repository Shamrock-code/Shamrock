// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FaceFlagger.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/SourceLocation.hpp"
#include "shammodels/zeus/NeighFaceList.hpp"
#include "shammodels/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

// this flags faces but not your face

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::FaceFlagger<Tvec, TgridVec>::flag_faces() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<sycl::buffer<u8>> face_normals_dat_lookup;

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchDataLayer &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        tree::ObjectCache &pcache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> face_normals_lookup(pcache.sum_neigh_cnt);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;
        auto cell_min   = buf_cell_min.get_read_access(depends_list);
        auto cell_max   = buf_cell_max.get_read_access(depends_list);
        auto cloop_ptrs = pcache.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            tree::ObjectCacheIterator cell_looper(cloop_ptrs);

            sycl::accessor normals_lookup{
                face_normals_lookup, cgh, sycl::write_only, sycl::no_init};

            shambase::parallel_for(cgh, mpdat.total_elements, "flag_neigh", [=](u64 id_a) {
                TgridVec cell2_a = (cell_min[id_a] + cell_max[id_a]);

                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u64 id_list) {
                    TgridVec cell2_b = (cell_min[id_b] + cell_max[id_b]);
                    TgridVec cell2_d = cell2_b - cell2_a;

                    TgridVec d_norm = sycl::abs(cell2_d).template convert<Tgridscal>();

                    // I mean if you are up to such
                    Tgridscal max_compo = sycl::max(sycl::max(d_norm.x(), d_norm.y()), d_norm.z());

                    // what a readable piece of code
                    // there can be only ONE that is the true answers
                    const u8 lookup = ((cell2_d.x() == -max_compo) ? 0 : 0)
                                      + ((cell2_d.x() == max_compo) ? 1 : 0)
                                      + ((cell2_d.y() == -max_compo) ? 2 : 0)
                                      + ((cell2_d.y() == max_compo) ? 3 : 0)
                                      + ((cell2_d.z() == -max_compo) ? 4 : 0)
                                      + ((cell2_d.z() == max_compo) ? 5 : 0);

                    // if(cell_min[id_a].x() < 0 && cell_min[id_a].y() == 10){
                    //     sycl::ext::oneapi::experimental::printf("%d (%ld %ld %ld) : %d\n", id_a,
                    //     cell_min[id_a].x(),cell_min[id_a].y(),cell_min[id_a].z()
                    //     ,u32(lookup));
                    // }

                    // F this bit bit of code
                    // i'm so done with this crap
                    // godbolts gods command's you to inline !
                    normals_lookup[id_list] = lookup;
                });

                // Chaptgpt beautifull poem about the beautifullness of the SIMD instructions
                //
                // Oh, SIMD instructions, you tangled mess,
                // A source of frustration, I must confess.
                // You promised speed, you boasted grace,
                // Yet you leave my code a tangled case.
                //
                // With your cryptic syntax and obscure ways,
                // You lead me into a bewildering maze.
                // I try to optimize, to harness your might,
                // But your convoluted logic gives me a fright.
                //
                // You claim to be efficient, a boon to behold,
                // Yet your pitfalls and traps leave me cold.
                // I chase after vectors, I chase after speed,
                // But your complexities multiply with every need.
                //
                // Oh, SIMD instructions, you deceptive charm,
                // You leave my patience disarmed.
                // A Pandora's box of headaches and woes,
                // In your shadowy realm, my confidence slows.
                //
                // So here's to you, SIMD, with a bitter disdain,
                // Your alluring facade hides nothing but pain.
                // You promised elegance, you promised glee,
                // But all I find is chaos, as you laugh at me.
            });
        });

        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        pcache.complete_event_state(resulting_events);

        // store the buffer in distrib data
        face_normals_dat_lookup.add_obj(p.id_patch, std::move(face_normals_lookup));
    });

    storage.face_normals_lookup.set(std::move(face_normals_dat_lookup));
}

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::FaceFlagger<Tvec, TgridVec>::split_face_list() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shambase::DistributedData<NeighFaceList<Tvec>> neigh_lst;

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchDataLayer &pdat) {
        shamrock::tree::ObjectCache &cache = storage.neighbors_cache.get().get_cache(p.id_patch);

        sycl::buffer<u8> &face_normals_lookup = storage.face_normals_lookup.get().get(p.id_patch);

        auto build_flist = [&](u8 lookup) -> OrientedNeighFaceList<Tvec> {
            return {isolate_lookups(cache, face_normals_lookup, lookup), lookup_to_normal(lookup)};
        };

        auto build_neigh_list = [&]() -> NeighFaceList<Tvec> {
            return {
                build_flist(0),
                build_flist(1),
                build_flist(2),
                build_flist(3),
                build_flist(4),
                build_flist(5)};
        };

        neigh_lst.add_obj(p.id_patch, build_neigh_list());
    });

    storage.neighbors_cache.reset();
    storage.face_normals_lookup.reset();

    storage.face_lists.set(std::move(neigh_lst));
}

struct AMRNeighIds {
    // since it's AMR with only delta = 1 in level
    // only cases are :
    //  - same level : block_id <-> block_id map
    //  - increase level :  block_id <-> block_id map
    //  - decrease level : block_id <-> block_id + divcoord map
    //          divcoord are to see in which suboct of
    //          the block the neigh is in.

    u64 id_patch;

    struct {

        sycl::buffer<u32> block_ids;

    } level_p1;

    struct {

        sycl::buffer<u32> block_ids;

        sycl::buffer<u32> cell_xm;
        sycl::buffer<u32> cell_xp;
        sycl::buffer<u32> cell_ym;
        sycl::buffer<u32> cell_yp;
        sycl::buffer<u32> cell_zm;
        sycl::buffer<u32> cell_zp;

    } level_m1;

    struct {

        // ids of the blocks having
        sycl::buffer<u32> block_ids;

        // neigh[block_id*block_size + cell_id]
        //     -> neighbourgh cell (block_id*block_size + cell_id)
        sycl::buffer<u32> cell_xm;
        sycl::buffer<u32> cell_xp;
        sycl::buffer<u32> cell_ym;
        sycl::buffer<u32> cell_yp;
        sycl::buffer<u32> cell_zm;
        sycl::buffer<u32> cell_zp;

    } level_same;
};

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::FaceFlagger<Tvec, TgridVec>::compute_neigh_ids() {}

template<class Tvec, class TgridVec>
shamrock::tree::ObjectCache shammodels::zeus::modules::FaceFlagger<Tvec, TgridVec>::isolate_lookups(
    shamrock::tree::ObjectCache &cache, sycl::buffer<u8> &face_normals_lookup, u8 lookup_value) {

    u32 obj_cnt = cache.cnt_neigh.get_size();

    sham::DeviceBuffer<u32> face_count(obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

    {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto cloop_ptrs = cache.get_read_access(depends_list);
        auto face_cnts  = face_count.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator cell_looper(cloop_ptrs);

            sycl::accessor normals_lookup{face_normals_lookup, cgh, sycl::read_only};

            u8 wanted_lookup = lookup_value;

            shambase::parallel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32) gid;

                u32 cnt = 0;
                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u32 id_list) {
                    cnt += (normals_lookup[id_list] == wanted_lookup) ? 1 : 0;
                });

                face_cnts[id_a] = cnt;
            });
        });

        sham::EventList resulting_events;
        resulting_events.add_event(e);

        cache.complete_event_state(resulting_events);
        face_count.complete_event_state(resulting_events);
    }

    shamrock::tree::ObjectCache pcache
        = shamrock::tree::prepare_object_cache(std::move(face_count), obj_cnt);

    shamsys::instance::get_compute_queue().wait();

    {
        NamedStackEntry stack_loc2{"fill cache"};
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto cloop_ptrs        = cache.get_read_access(depends_list);
        auto scanned_neigh_cnt = pcache.scanned_cnt.get_read_access(depends_list);
        auto neigh             = pcache.index_neigh_map.get_write_access(depends_list);

        // logger::raw_ln(obj_cnt, pcache.cnt_neigh.size(),pcache.scanned_cnt.size(),
        // pcache.index_neigh_map.size());

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectCacheIterator cell_looper(cloop_ptrs);

            sycl::accessor normals_lookup{face_normals_lookup, cgh, sycl::read_only};

            u8 wanted_lookup = lookup_value;

            shambase::parallel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32) gid;
                u32 cnt  = scanned_neigh_cnt[id_a];

                // sycl::ext::oneapi::experimental::printf("%d %d\n", id_a,cnt);

                cell_looper.for_each_object_with_id(id_a, [&](u32 id_b, u32 id_list) {
                    bool lookup_match = normals_lookup[id_list] == wanted_lookup;
                    if (lookup_match) {
                        // sycl::ext::oneapi::experimental::printf("%d %d %d %d\n",
                        // id_a,cnt,id_b,id_list);
                        neigh[cnt] = id_b;
                        cnt++;
                    }
                });
            });
        });

        pcache.scanned_cnt.complete_event_state(e);
        pcache.index_neigh_map.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        cache.complete_event_state(resulting_events);
    }

    shamlog_debug_sycl_ln(
        "AMR::FaceFlagger", "lookup :", lookup_value, "found N =", pcache.sum_neigh_cnt);

    return pcache;
}

template class shammodels::zeus::modules::FaceFlagger<f64_3, i64_3>;
