// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHSolverImpl.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/SPHSolverImpl.hpp"

shamrock::tree::ObjectCache shammodels::SPHSolverImpl::build_neigh_cache(
    u32 start_offset,
    u32 obj_cnt,
    sycl::buffer<vec> &buf_xyz,
    sycl::buffer<flt> &buf_hpart,
    RadixTree<u_morton, vec> &tree,
    sycl::buffer<flt> &tree_field_hmax) {

    StackEntry stack_loc{};

    using namespace shamrock;

    sham::DeviceBuffer<u32> neigh_count(obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

    {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;
        auto neigh_cnt = neigh_count.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, start_offset](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr flt Rker2 = Kernel::Rkern * Kernel::Rkern;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                u32 id_a = start_offset + (u32) item.get_id(0);

                flt h_a = hpart[id_a];

                vec xyz_a = xyz[id_a];

                vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, vec bmin, vec bmax) -> bool {
                        flt int_r_max_cell = hmax_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        vec dr   = xyz_a - xyz[id_b];
                        flt rab2 = sycl::dot(dr, dr);
                        flt h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        cnt++;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        neigh_count.complete_event_state(e);
    }

    tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);
    {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;
        auto scanned_neigh_cnt = pcache.scanned_cnt.get_read_access(depends_list);
        auto neigh             = pcache.index_neigh_map.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, start_offset](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor hmax_tree{tree_field_hmax, cgh, sycl::read_only};

            // sycl::stream out {4096,1024,cgh};

            constexpr flt Rker2 = Kernel::Rkern * Kernel::Rkern;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                u32 id_a = start_offset + (u32) item.get_id(0);

                flt h_a = hpart[id_a];

                vec xyz_a = xyz[id_a];

                vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, vec bmin, vec bmax) -> bool {
                        flt int_r_max_cell = hmax_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        vec dr   = xyz_a - xyz[id_b];
                        flt rab2 = sycl::dot(dr, dr);
                        flt h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }
                        neigh[cnt] = id_b;
                        cnt++;
                    });
            });
        });
        pcache.scanned_cnt.complete_event_state(e);
        pcache.index_neigh_map.complete_event_state(e);
    }

    return pcache;
}

shamrock::tree::ObjectCache shammodels::SPHSolverImpl::build_hiter_neigh_cache(
    u32 start_offset,
    u32 obj_cnt,
    sycl::buffer<vec> &buf_xyz,
    sycl::buffer<flt> &buf_hpart,
    RadixTree<u_morton, vec> &tree,
    flt h_tolerance) {

    StackEntry stack_loc{};

    using namespace shamrock;

    sham::DeviceBuffer<u32> neigh_count(obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

    {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;
        auto neigh_cnt = neigh_count.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, start_offset, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            constexpr flt Rker2 = Kernel::Rkern * Kernel::Rkern;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                u32 id_a = start_offset + (u32) item.get_id(0);
                // increase smoothing length to include possible future neigh in the cache
                flt h_a  = hpart[id_a] * h_tolerance;
                flt dint = h_a * h_a * Rker2;

                vec xyz_a = xyz[id_a];

                vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32, vec bmin, vec bmax) -> bool {
                        return shammath::domain_are_connected(
                            bmin, bmax, inter_box_a_min, inter_box_a_max);
                    },
                    [&](u32 id_b) {
                        vec dr   = xyz_a - xyz[id_b];
                        flt rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        cnt++;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });
        neigh_count.complete_event_state(e);
    }

    tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

    {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;
        auto scanned_neigh_cnt = pcache.scanned_cnt.get_read_access(depends_list);
        auto neigh             = pcache.index_neigh_map.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, start_offset, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            constexpr flt Rker2 = Kernel::Rkern * Kernel::Rkern;

            cgh.parallel_for(sycl::range<1>{obj_cnt}, [=](sycl::item<1> item) {
                u32 id_a = start_offset + (u32) item.get_id(0);

                // increase smoothing length to include possible future neigh in the cache
                flt h_a  = hpart[id_a] * h_tolerance;
                flt dint = h_a * h_a * Rker2;

                vec xyz_a = xyz[id_a];

                vec inter_box_a_min = xyz_a - h_a * Kernel::Rkern;
                vec inter_box_a_max = xyz_a + h_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32, vec bmin, vec bmax) -> bool {
                        return shammath::domain_are_connected(
                            bmin, bmax, inter_box_a_min, inter_box_a_max);
                    },
                    [&](u32 id_b) {
                        vec dr   = xyz_a - xyz[id_b];
                        flt rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }
                        neigh[cnt] = id_b;
                        cnt++;
                    });
            });
        });
        pcache.scanned_cnt.complete_event_state(e);
        pcache.index_neigh_map.complete_event_state(e);
    }

    return pcache;
}
