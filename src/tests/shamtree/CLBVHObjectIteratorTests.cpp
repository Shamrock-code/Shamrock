// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/CLBVHObjectIterator.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <vector>

using Tmorton = u64;
using Tvec    = f64_3;
using Tscal   = shambase::VecComponent<Tvec>;

TestStart(Unittest, "shamtree/LCBVHObjectIterator", test_lcbvh_object_iterator, 1) {

    std::vector<Tvec> partpos{
        Tvec(0, 0, 0),
        Tvec(0.1, 0.0, 0.0),
        Tvec(0.0, 0.1, 0.0),
        Tvec(0.0, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.0),
        Tvec(0.0, 0.1, 0.1),
        Tvec(0.1, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.1),
        Tvec(0.2, 0.2, 0.2),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4),
        Tvec(1, 1, 1),
        Tvec(2, 2, 2),
        Tvec(-1, -1, -1)};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(partpos.size(), dev_sched);

    partpos_buf.copy_from_stdvec(partpos);

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    bvh.rebuild_from_positions(partpos_buf, bb, 1);

    REQUIRE_EQUAL(bvh.structure.get_internal_cell_count(), 6);

    auto obj_it = bvh.get_object_iterator();

    { // find everything

        std::vector<u32> expected_counts = {14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14};
        std::vector<u32> expected_neigh  = {
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
        };

        sham::DeviceBuffer<u32> cnt_buf(partpos.size(), dev_sched);
        cnt_buf.fill(0);

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it},
            sham::MultiRef{cnt_buf},
            partpos.size(),
            [](u32 i, auto obj_it, auto cnt) {
                u32 counter = 0;
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return true;
                    },
                    [&](u32 obj_id) {
                        counter++;
                    });
                cnt[i] = counter;
            });

        REQUIRE_EQUAL(cnt_buf.copy_to_stdvec(), expected_counts);

        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(cnt_buf), partpos_buf.get_size());

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, pcache.scanned_cnt},
            sham::MultiRef{pcache.index_neigh_map},
            partpos.size(),
            [](u32 i, auto obj_it, const u32 *scanned_cnt, u32 *index_neigh_map) {
                u32 counter = scanned_cnt[i];
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return true;
                    },
                    [&](u32 obj_id) {
                        bool no_interact = false;
                        if (!no_interact) {
                            index_neigh_map[counter] = obj_id;
                        }
                        counter += (no_interact) ? 0 : 1;
                    });
            });

        REQUIRE_EQUAL(pcache.index_neigh_map.copy_to_stdvec(), expected_neigh);
    }

    { // find within a box around particles

        std::vector<u32> expected_counts = {
            8, // (0, 0, 0),
            8, // (0.1, 0.0, 0.0),
            8, // (0.0, 0.1, 0.0),
            8, // (0.0, 0.0, 0.1),
            8, // (0.1, 0.1, 0.0),
            8, // (0.0, 0.1, 0.1),
            8, // (0.1, 0.0, 0.1),
            9, // (0.1, 0.1, 0.1),
            3, // (0.2, 0.2, 0.2),
            3, // (0.3, 0.3, 0.3),
            2, // (0.4, 0.4, 0.4),
            1, // (1, 1, 1),
            1, // (2, 2, 2),
            1  // (-1, -1, -1)
        };
        std::vector<u32> expected_neigh = {
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7, 8, //
            7,  8,  9,                    //
            8,  9,  10,                   //
            9,  10,                       //
            11,                           //
            12,                           //
            13                            //
        };

        sham::DeviceBuffer<u32> cnt_buf(partpos.size(), dev_sched);
        cnt_buf.fill(0);

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, partpos_buf},
            sham::MultiRef{cnt_buf},
            partpos.size(),
            [](u32 i, auto obj_it, const Tvec *pos, auto cnt) {
                Tvec r = pos[i];

                Tscal s = 0.15;

                shammath::AABB<Tvec> test_aabb(r + Tvec{-s, -s, -s}, r + Tvec{s, s, s});

                u32 counter = 0;
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return node_aabb.get_intersect(test_aabb).is_not_empty();
                    },
                    [&](u32 obj_id) {
                        Tvec r2 = pos[obj_id];
                        shammath::AABB<Tvec> test_aabb2(r2, r2);
                        if (test_aabb2.get_intersect(test_aabb).is_not_empty()) {
                            counter++;
                        }
                    });
                cnt[i] = counter;
            });

        REQUIRE_EQUAL(cnt_buf.copy_to_stdvec(), expected_counts);

        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(cnt_buf), partpos_buf.get_size());

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, partpos_buf, pcache.scanned_cnt},
            sham::MultiRef{pcache.index_neigh_map},
            partpos.size(),
            [](u32 i, auto obj_it, const Tvec *pos, const u32 *scanned_cnt, u32 *index_neigh_map) {
                Tvec r = pos[i];

                Tscal s = 0.15;

                shammath::AABB<Tvec> test_aabb(r + Tvec{-s, -s, -s}, r + Tvec{s, s, s});

                u32 counter = scanned_cnt[i];
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return node_aabb.get_intersect(test_aabb).is_not_empty();
                    },
                    [&](u32 obj_id) {
                        Tvec r2 = pos[obj_id];
                        shammath::AABB<Tvec> test_aabb2(r2, r2);
                        if (test_aabb2.get_intersect(test_aabb).is_not_empty()) {

                            bool no_interact = false;
                            if (!no_interact) {
                                index_neigh_map[counter] = obj_id;
                            }
                            counter += (no_interact) ? 0 : 1;
                        }
                    });
            });

        REQUIRE_EQUAL(pcache.index_neigh_map.copy_to_stdvec(), expected_neigh);
    }
}

TestStart(
    Unittest, "shamtree/LCBVHObjectIterator(one-cell)", test_lcbvh_object_iterator_one_cell, 1) {

    std::vector<Tvec> partpos{
        Tvec(0, 0, 0),
        Tvec(0.1, 0.0, 0.0),
        Tvec(0.0, 0.1, 0.0),
        Tvec(0.0, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.0),
        Tvec(0.0, 0.1, 0.1),
        Tvec(0.1, 0.0, 0.1),
        Tvec(0.1, 0.1, 0.1),
        Tvec(0.2, 0.2, 0.2),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4),
        Tvec(1, 1, 1),
        Tvec(2, 2, 2),
        Tvec(-1, -1, -1)};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto &q        = dev_sched->get_queue();

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(partpos.size(), dev_sched);

    partpos_buf.copy_from_stdvec(partpos);

    auto bvh = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>::make_empty(dev_sched);

    bvh.rebuild_from_positions(partpos_buf, bb, 8);

    REQUIRE_EQUAL(bvh.structure.get_internal_cell_count(), 0);

    auto obj_it = bvh.get_object_iterator();

    { // find everything

        std::vector<u32> expected_counts = {14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14};
        std::vector<u32> expected_neigh  = {
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
            13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, //
        };

        sham::DeviceBuffer<u32> cnt_buf(partpos.size(), dev_sched);
        cnt_buf.fill(0);

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it},
            sham::MultiRef{cnt_buf},
            partpos.size(),
            [](u32 i, auto obj_it, auto cnt) {
                u32 counter = 0;
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return true;
                    },
                    [&](u32 obj_id) {
                        counter++;
                    });
                cnt[i] = counter;
            });

        REQUIRE_EQUAL(cnt_buf.copy_to_stdvec(), expected_counts);

        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(cnt_buf), partpos_buf.get_size());

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, pcache.scanned_cnt},
            sham::MultiRef{pcache.index_neigh_map},
            partpos.size(),
            [](u32 i, auto obj_it, const u32 *scanned_cnt, u32 *index_neigh_map) {
                u32 counter = scanned_cnt[i];
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return true;
                    },
                    [&](u32 obj_id) {
                        bool no_interact = false;
                        if (!no_interact) {
                            index_neigh_map[counter] = obj_id;
                        }
                        counter += (no_interact) ? 0 : 1;
                    });
            });

        REQUIRE_EQUAL(pcache.index_neigh_map.copy_to_stdvec(), expected_neigh);
    }

    { // find within a box around particles

        std::vector<u32> expected_counts = {
            8, // (0, 0, 0),
            8, // (0.1, 0.0, 0.0),
            8, // (0.0, 0.1, 0.0),
            8, // (0.0, 0.0, 0.1),
            8, // (0.1, 0.1, 0.0),
            8, // (0.0, 0.1, 0.1),
            8, // (0.1, 0.0, 0.1),
            9, // (0.1, 0.1, 0.1),
            3, // (0.2, 0.2, 0.2),
            3, // (0.3, 0.3, 0.3),
            2, // (0.4, 0.4, 0.4),
            1, // (1, 1, 1),
            1, // (2, 2, 2),
            1  // (-1, -1, -1)
        };
        std::vector<u32> expected_neigh = {
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7,    //
            0,  3,  2,  5, 1, 6, 4, 7, 8, //
            7,  8,  9,                    //
            8,  9,  10,                   //
            9,  10,                       //
            11,                           //
            12,                           //
            13                            //
        };

        sham::DeviceBuffer<u32> cnt_buf(partpos.size(), dev_sched);
        cnt_buf.fill(0);

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, partpos_buf},
            sham::MultiRef{cnt_buf},
            partpos.size(),
            [](u32 i, auto obj_it, const Tvec *pos, auto cnt) {
                Tvec r = pos[i];

                Tscal s = 0.15;

                shammath::AABB<Tvec> test_aabb(r + Tvec{-s, -s, -s}, r + Tvec{s, s, s});

                u32 counter = 0;
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return node_aabb.get_intersect(test_aabb).is_not_empty();
                    },
                    [&](u32 obj_id) {
                        Tvec r2 = pos[obj_id];
                        shammath::AABB<Tvec> test_aabb2(r2, r2);
                        if (test_aabb2.get_intersect(test_aabb).is_not_empty()) {
                            counter++;
                        }
                    });
                cnt[i] = counter;
            });

        REQUIRE_EQUAL(cnt_buf.copy_to_stdvec(), expected_counts);

        shamrock::tree::ObjectCache pcache
            = shamrock::tree::prepare_object_cache(std::move(cnt_buf), partpos_buf.get_size());

        sham::kernel_call(
            q,
            sham::MultiRef{obj_it, partpos_buf, pcache.scanned_cnt},
            sham::MultiRef{pcache.index_neigh_map},
            partpos.size(),
            [](u32 i, auto obj_it, const Tvec *pos, const u32 *scanned_cnt, u32 *index_neigh_map) {
                Tvec r = pos[i];

                Tscal s = 0.15;

                shammath::AABB<Tvec> test_aabb(r + Tvec{-s, -s, -s}, r + Tvec{s, s, s});

                u32 counter = scanned_cnt[i];
                obj_it.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return node_aabb.get_intersect(test_aabb).is_not_empty();
                    },
                    [&](u32 obj_id) {
                        Tvec r2 = pos[obj_id];
                        shammath::AABB<Tvec> test_aabb2(r2, r2);
                        if (test_aabb2.get_intersect(test_aabb).is_not_empty()) {

                            bool no_interact = false;
                            if (!no_interact) {
                                index_neigh_map[counter] = obj_id;
                            }
                            counter += (no_interact) ? 0 : 1;
                        }
                    });
            });

        REQUIRE_EQUAL(pcache.index_neigh_map.copy_to_stdvec(), expected_neigh);
    }
}
