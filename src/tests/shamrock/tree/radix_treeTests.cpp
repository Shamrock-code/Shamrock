// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "TreeTests.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/RadixTree.hpp"
#include <vector>

#if false


Test_start("radix_tree",test_new_pfield_compute,1){
    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;
    using vec3i = morton_3d::morton_types<morton_mode>::int_vec_repr;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);
    pdl.add_field<flt>("h", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");
    const auto id_h = pdl.get_field_idx<flt>("h");

    PatchData pdat(pdl);
    pdat.resize(npart);



    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        auto & h_part = pdat.get_field<flt>(id_h).get_buf();
        sycl::host_accessor<flt> h {*h_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
            h[i] = 0.2 + distf(eng)*0.1;
        }
    }


    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            shamsys::instance::get_compute_queue(),
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(),
            npart , reduc_level
        );

    rtree.compute_cellvolume(shamsys::instance::get_compute_queue());


    flt h_tol = 1.2;

    auto h_max = rtree.compute_field<flt>(
        shamsys::instance::get_compute_queue(),
        1,
        [&](sycl::handler &cgh,auto && node_looper){

            auto & h_part = pdat.get_field<flt>(id_h).get_buf();
            auto h = sycl::accessor{* h_part, cgh, sycl::read_only};

            node_looper(
                [=](auto && particle_looper, auto & buf, auto && get_id_store){
                    flt h_tmp = -1;

                    particle_looper([&](u32 particle_id){
                        f32 h_a = h[particle_id] * h_tol;
                        h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                    });

                    buf[get_id_store()] = h_tmp;
                }
            );

        },
        [&](auto && get_left_val, auto && get_right_val, auto & buf, auto && get_id_store){
            flt h_l = get_left_val(0);
            flt h_r = get_right_val(0);

            buf[get_id_store()] = (h_r > h_l ? h_r : h_l) ;
        }
    );





    auto compute_old = [&]() -> auto {
        shamlog_debug_sycl_ln("RadixTree", "compute int boxes");

        auto buf_cell_interact_rad = std::make_unique<sycl::buffer<flt>>(rtree.tree_internal_count + rtree.tree_leaf_count);
        sycl::range<1> range_leaf_cell{rtree.tree_leaf_count};

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            u32 offset_leaf = rtree.tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::discard_write>(cgh);
            auto h          = pdat.get_field<flt>(id_h).get_buf()->template get_access<sycl::access::mode::read>(cgh);

            auto cell_particle_ids  = rtree.buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
            auto particle_index_map = rtree.buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);

            flt tol = h_tol;

            cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 min_ids = cell_particle_ids[gid];
                u32 max_ids = cell_particle_ids[gid + 1];
                f32 h_tmp   = 0;

                for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                    f32 h_a = h[particle_index_map[id_s]] * tol;
                    h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                }

                h_max_cell[offset_leaf + gid] = h_tmp;
            });
        });

        sycl::range<1> range_tree{rtree.tree_internal_count};
        auto ker_reduc_hmax = [&](sycl::handler &cgh) {
            u32 offset_leaf = rtree.tree_internal_count;

            auto h_max_cell = buf_cell_interact_rad->template get_access<sycl::access::mode::read_write>(cgh);

            auto rchild_id   = rtree.buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
            auto lchild_id   = rtree.buf_lchild_id->get_access<sycl::access::mode::read>(cgh);
            auto rchild_flag = rtree.buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
            auto lchild_flag = rtree.buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                flt h_l = h_max_cell[lid];
                flt h_r = h_max_cell[rid];

                h_max_cell[gid] = (h_r > h_l ? h_r : h_l);
            });
        };

        for (u32 i = 0; i < rtree.tree_depth; i++) {
            shamsys::instance::get_compute_queue().submit(ker_reduc_hmax);
        }

        return std::move(buf_cell_interact_rad);
    };


    {
        auto old_vers = compute_old();

        u32 len = old_vers->size();

        sycl::host_accessor acc_old {*old_vers};
        sycl::host_accessor acc_new {*h_max.radix_tree_field_buf};

        for(u32 i = 0 ; i < len; i ++){
            Test_assert("same_buf", test_sycl_eq(acc_new[i],acc_old[i]));
        }
    }


}


constexpr auto list_npart_test = {1.14975700e+02,
       1.35304777e+02, 1.59228279e+02, 1.87381742e+02, 2.20513074e+02,
       2.59502421e+02, 3.05385551e+02, 3.59381366e+02, 4.22924287e+02,
       4.97702356e+02, 5.85702082e+02, 6.89261210e+02, 8.11130831e+02,
       9.54548457e+02, 1.12332403e+03, 1.32194115e+03, 1.55567614e+03,
       1.83073828e+03, 2.15443469e+03, 2.53536449e+03, 2.98364724e+03,
       3.51119173e+03, 4.13201240e+03, 4.86260158e+03, 5.72236766e+03,
       6.73415066e+03, 7.92482898e+03, 9.32603347e+03, 1.09749877e+04,
       1.29154967e+04, 1.51991108e+04, 1.78864953e+04, 2.10490414e+04,
       2.47707636e+04, 2.91505306e+04, 3.43046929e+04, 4.03701726e+04,
       4.75081016e+04, 5.59081018e+04, 6.57933225e+04, 7.74263683e+04,
       9.11162756e+04, 1.07226722e+05, 1.26185688e+05, 1.48496826e+05,
       1.74752840e+05, 2.05651231e+05, 2.42012826e+05, 2.84803587e+05,
       3.35160265e+05, 3.94420606e+05, 4.64158883e+05, 5.46227722e+05,
       6.42807312e+05, 7.56463328e+05, 8.90215085e+05, 1.04761575e+06,
       1.23284674e+06, 1.45082878e+06, 1.70735265e+06, 2.00923300e+06,
       2.36448941e+06, 2.78255940e+06, 3.27454916e+06, 3.85352859e+06,
       4.53487851e+06, 5.33669923e+06, 6.28029144e+06, 7.39072203e+06,
       8.69749003e+06, 1.02353102e+07};

       //, 1.20450354e+07, 1.41747416e+07,
       //1.66810054e+07, 1.96304065e+07, 2.31012970e+07, 2.71858824e+07,
       //3.19926714e+07, 3.76493581e+07, 4.43062146e+07, 5.21400829e+07,
       //6.13590727e+07, 7.22080902e+07, 8.49753436e+07, 1.00000000e+08};


Bench_start("tree field old compute performance", "treefieldcomputeperf_new", treefieldcomputeperf_new, 1){

    auto run_bench = [&](u32 npart, u32 reduc_level){

        using flt = f32;
        using vec = sycl::vec<flt,3>;
        using morton_mode = u32;
        using vec3i = morton_3d::morton_types<morton_mode>::int_vec_repr;

        std::mt19937 eng(0x1111);
        std::uniform_real_distribution<flt> distf(-1, 1);


        PatchDataLayout pdl;
        pdl.add_field<vec>("xyz", 1);
        pdl.add_field<flt>("h", 1);

        const auto id_xyz = pdl.get_field_idx<vec>("xyz");
        const auto id_h = pdl.get_field_idx<flt>("h");

        PatchData pdat(pdl);
        pdat.resize(npart);



        {
            auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
            sycl::host_accessor<vec> pos {*pos_part};

            auto & h_part = pdat.get_field<flt>(id_h).get_buf();
            sycl::host_accessor<flt> h {*h_part};

            for (u32 i = 0; i < npart; i ++) {
                pos[i] = vec{distf(eng), distf(eng), distf(eng)};
                h[i] = 0.2 + distf(eng)*0.1;
            }
        }



        auto rtree = Radix_Tree<morton_mode, vec>(
                shamsys::instance::get_compute_queue(),
                {vec{-1,-1,-1},vec{1,1,1}},
                pdat.get_field<vec>(id_xyz).get_buf(),
                npart , reduc_level
            );

        rtree.compute_cellvolume(shamsys::instance::get_compute_queue());


        flt h_tol = 1.2;

        shamsys::instance::get_compute_queue().wait();

        Timer timer; timer.start();

        auto h_max = rtree.compute_field<flt>(
            shamsys::instance::get_compute_queue(),
            1,
            [&](sycl::handler &cgh,auto && node_looper){

                auto & h_part = pdat.get_field<flt>(id_h).get_buf();
                auto h = sycl::accessor{* h_part, cgh, sycl::read_only};

                node_looper(
                    [=](auto && particle_looper, auto & buf, auto && get_id_store){
                        flt h_tmp = -1;

                        particle_looper([&](u32 particle_id){
                            f32 h_a = h[particle_id] * h_tol;
                            h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
                        });

                        buf[get_id_store()] = h_tmp;
                    }
                );

            },
            [&](auto && get_left_val, auto && get_right_val, auto & buf, auto && get_id_store){
                flt h_l = get_left_val(0);
                flt h_r = get_right_val(0);

                buf[get_id_store()] = (h_r > h_l ? h_r : h_l) ;
            }
        );


        shamsys::instance::get_compute_queue().wait();

        timer.end();

        Register_score("%result = " + std::to_string(npart) + "," + std::to_string(timer.nanosec));

    };


    for(u32 i = 0;i < 6; i++){

        Register_score("%tree_reduc = " + std::to_string(i));
        for(u32 cnt : list_npart_test){
            run_bench(cnt,i);
        }

    }


}













template<class flt, class morton_mode>
void test_tree_comm(TestResults &__test_result_ref){

    using vec = sycl::vec<flt,3>;
    using vec3i = typename morton_3d::morton_types<morton_mode>::int_vec_repr;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");

    PatchData pdat(pdl);
    pdat.resize(npart);

    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
        }
    }


    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            shamsys::instance::get_compute_queue(),
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(),
            npart , reduc_level
        );

    rtree.compute_cellvolume(shamsys::instance::get_compute_queue());





    if(shamcomm::world_rank() == 0){
        std::vector<tree_comm::RadixTreeMPIRequest<morton_mode, vec>> rqs;
        tree_comm::comm_isend(rtree, rqs, 1, 0, MPI_COMM_WORLD);
        tree_comm::wait_all(rqs);
    }

    if(shamcomm::world_rank() == 1){
        std::vector<tree_comm::RadixTreeMPIRequest<morton_mode, vec>> rqs;

        auto rtree_recv = Radix_Tree<morton_mode, vec>::make_empty();

        tree_comm::comm_irecv_probe(rtree_recv, rqs, 0, 0, MPI_COMM_WORLD);
        tree_comm::wait_all(rqs);

        using t = sycl::vec<float, 3>::element_type;

        Test_assert("",rtree.one_cell_mode == rtree_recv.one_cell_mode);
        Test_assert("",test_sycl_eq(std::get<0>(rtree.box_coord) , std::get<0>(rtree_recv.box_coord)));
        Test_assert("",test_sycl_eq(std::get<1>(rtree.box_coord) , std::get<1>(rtree_recv.box_coord)));
        Test_assert("",rtree.obj_cnt == rtree_recv.obj_cnt);
        Test_assert("",rtree.tree_leaf_count == rtree_recv.tree_leaf_count);
        Test_assert("",rtree.tree_internal_count == rtree_recv.tree_internal_count);


        Test_assert("",syclalgs::reduction::equals(*rtree.buf_morton, *rtree_recv.buf_morton, rtree.obj_cnt));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_particle_index_map, *rtree_recv.buf_particle_index_map, rtree.obj_cnt));

        Test_assert("",syclalgs::reduction::equals(*rtree.buf_reduc_index_map, *rtree_recv.buf_reduc_index_map, rtree.tree_leaf_count+1));

        Test_assert("",syclalgs::reduction::equals(*rtree.buf_tree_morton, *rtree_recv.buf_tree_morton, rtree.tree_leaf_count));


        Test_assert("",syclalgs::reduction::equals(*rtree.buf_lchild_id, *rtree_recv.buf_lchild_id, rtree.tree_internal_count));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_rchild_id, *rtree_recv.buf_rchild_id, rtree.tree_internal_count));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_lchild_flag, *rtree_recv.buf_lchild_flag, rtree.tree_internal_count));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_rchild_flag, *rtree_recv.buf_rchild_flag, rtree.tree_internal_count));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_endrange, *rtree_recv.buf_endrange, rtree.tree_internal_count));

        Test_assert("",syclalgs::reduction::equals(*rtree.buf_pos_min_cell, *rtree_recv.buf_pos_min_cell, rtree.tree_internal_count + rtree.tree_leaf_count));
        Test_assert("",syclalgs::reduction::equals(*rtree.buf_pos_max_cell, *rtree_recv.buf_pos_max_cell, rtree.tree_internal_count + rtree.tree_leaf_count));
    }

}

Test_start("radix_tree", tree_comm, 2){

    test_tree_comm<f32,u32>(__test_result_ref);

}










Test_start("radix_tree", tree_cut, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1000;

    PatchDataLayout pdl;
    pdl.add_field<vec>("xyz", 1);

    const auto id_xyz = pdl.get_field_idx<vec>("xyz");

    PatchData pdat(pdl);
    pdat.resize(npart);



    {
        auto & pos_part = pdat.get_field<vec>(id_xyz).get_buf();
        sycl::host_accessor<vec> pos {*pos_part};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
        }
    }


    constexpr u32 reduc_level = 5;

    auto rtree = Radix_Tree<morton_mode, vec>(
            shamsys::instance::get_compute_queue(),
            {vec{-1,-1,-1},vec{1,1,1}},
            pdat.get_field<vec>(id_xyz).get_buf(),
            npart ,reduc_level
        );

    rtree.compute_cellvolume(shamsys::instance::get_compute_queue());




    sycl::buffer<u32> node_id_old = sycl::buffer<u32>(rtree.tree_internal_count+ rtree.tree_leaf_count);
    {
        sycl::host_accessor acc {node_id_old};
        for (u32 i = 0; i < rtree.tree_internal_count; i++) {
            acc[i] = i;
        }

        for (u32 i = 0; i < rtree.tree_leaf_count; i++) {
            acc[i + rtree.tree_internal_count] = i;
        }
    }


    {
        shamlog_debug_sycl_ln("Radixtree", "valid_node_state");
        rtree.print_tree_field(node_id_old);
        logger::raw_ln("");
    }





    std::tuple<vec, vec> cur_range{vec{-1,-1,-1},vec{0,0.5,1}};


    u32 total_count             = rtree.tree_internal_count + rtree.tree_leaf_count;
    sycl::range<1> range_tree{total_count};

    shamlog_debug_sycl_ln("Radixtree", "computing valid node buf");

    auto init_valid_buf = [&]() -> sycl::buffer<u8> {

        sycl::buffer<u8> valid_node = sycl::buffer<u8>(total_count);

        sycl::range<1> range_tree{total_count};

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc_valid_node{valid_node, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor acc_pos_cell_min{*rtree.buf_pos_min_cell_flt, cgh, sycl::read_only};
            sycl::accessor acc_pos_cell_max{*rtree.buf_pos_max_cell_flt, cgh, sycl::read_only};

            vec v_min = std::get<0>(cur_range);
            vec v_max = std::get<1>(cur_range);

            cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
                acc_valid_node[item] = BBAA::cella_neigh_b(v_min, v_max, acc_pos_cell_min[item], acc_pos_cell_max[item]);
            });
        });

        return valid_node;
    };



    std::unique_ptr<sycl::buffer<u8>> valid_buf = std::make_unique<sycl::buffer<u8>>(init_valid_buf());

    auto cut = rtree.cut_tree(shamsys::instance::get_compute_queue(), *valid_buf);

    valid_buf.reset();


}









Test_start("radix_tree", treeleveljump_cell_range_test, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::vector<vec> pos_part{
        vec{0,0,0},vec{0.05,0,0},
        vec{1,0,0},vec{0.95,0,0}
    };

    std::unique_ptr<sycl::buffer<vec>> xyz  = std::make_unique<sycl::buffer<vec>>(pos_part.data(),pos_part.size());


    auto rtree = Radix_Tree<morton_mode, vec>(
            shamsys::instance::get_compute_queue(),
            {vec{0,0,0},vec{1,1,1}},
            xyz,
            4 ,0
        );

    rtree.compute_cellvolume(shamsys::instance::get_compute_queue());

    {
        auto acc_min = sycl::host_accessor{*rtree.buf_pos_min_cell_flt};
        auto acc_max = sycl::host_accessor{*rtree.buf_pos_max_cell_flt};

        for(u32 i = 0 ; i < 7; i++){
            logger::raw_ln(i,acc_min[i],acc_max[i]);
        }
    }


}

#endif

template<class u_morton, class flt>
void test_inclusion(u32 Npart, u32 reduc_level) {
    using vec = sycl::vec<flt, 3>;

    auto coord_range = get_test_coord_ranges<vec>();

    auto pos = shamalgs::random::mock_buffer_ptr<vec>(
        0x111, Npart, coord_range.lower, coord_range.upper);

    RadixTree<u_morton, vec> rtree = RadixTree<u_morton, vec>(
        shamsys::instance::get_compute_queue(),
        {coord_range.lower, coord_range.upper},
        pos,
        Npart,
        reduc_level);

    rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());

    bool inclusion_valid = true;

    std::string comment;

    {
        sycl::host_accessor tree_acc_pos_min_cell{
            *rtree.tree_cell_ranges.buf_pos_min_cell, sycl::read_only};
        sycl::host_accessor tree_acc_pos_max_cell{
            *rtree.tree_cell_ranges.buf_pos_max_cell, sycl::read_only};
        u32 tree_leaf_offset = rtree.tree_struct.internal_cell_count;
        sycl::host_accessor tree_lchild_id{*rtree.tree_struct.buf_lchild_id, sycl::read_only};
        sycl::host_accessor tree_rchild_id{*rtree.tree_struct.buf_rchild_id, sycl::read_only};
        sycl::host_accessor tree_lchild_flag{*rtree.tree_struct.buf_lchild_flag, sycl::read_only};
        sycl::host_accessor tree_rchild_flag{*rtree.tree_struct.buf_rchild_flag, sycl::read_only};

        for (u32 i = 0; i < tree_leaf_offset; i++) {
            auto cur_pos_min_cell_a = tree_acc_pos_min_cell[i];
            auto cur_pos_max_cell_a = tree_acc_pos_max_cell[i];

            auto inclusion_crit = [&](auto other_min, auto other_max) -> bool {
                return (cur_pos_min_cell_a.x() <= other_min.x())
                       && (cur_pos_min_cell_a.y() <= other_min.y())
                       && (cur_pos_min_cell_a.z() <= other_min.z())
                       && (cur_pos_max_cell_a.x() >= other_max.x())
                       && (cur_pos_max_cell_a.y() >= other_max.y())
                       && (cur_pos_max_cell_a.z() >= other_max.z());
            };

            u32 lid = tree_lchild_id[i] + tree_leaf_offset * tree_lchild_flag[i];
            u32 rid = tree_rchild_id[i] + tree_leaf_offset * tree_rchild_flag[i];

            // because the rid is volontarly modified in one cell mode
            // if(rtree.tree_struct.one_cell_mode){
            //     rid = 2;
            // }

            auto cur_pos_min_cell_bl = tree_acc_pos_min_cell[lid];
            auto cur_pos_max_cell_bl = tree_acc_pos_max_cell[lid];

            auto cur_pos_min_cell_br = tree_acc_pos_min_cell[rid];
            auto cur_pos_max_cell_br = tree_acc_pos_max_cell[rid];

            bool l_ok = inclusion_crit(cur_pos_min_cell_bl, cur_pos_max_cell_bl);
            bool r_ok = inclusion_crit(cur_pos_min_cell_br, cur_pos_max_cell_br);

            if (!(l_ok && r_ok)) {
                inclusion_valid = false;
                comment += shambase::format(
                    "fail : current : ({} {}) r:({} {}) l:({} {}) lid:{} rid:{}\n",
                    cur_pos_min_cell_a,
                    cur_pos_max_cell_a,
                    cur_pos_min_cell_bl,
                    cur_pos_max_cell_bl,
                    cur_pos_min_cell_br,
                    cur_pos_max_cell_br,
                    lid,
                    rid);
            }
        }
    }

    if (inclusion_valid) {
        REQUIRE_NAMED("inclusion ok", true);
    } else {
        shamtest::asserts().assert_add_comment(
            "inclusion ok",
            false,
            comment
                + shambase::format(
                    "\n leaf count : {}, internal count : {}",
                    rtree.tree_reduced_morton_codes.tree_leaf_count,
                    rtree.tree_struct.internal_cell_count));
    }
}

TestStart(
    Unittest, "shamrock/tree/RadixTree:bounding_volume_inclusion", bounding_volume_inclusion, 1) {
    test_inclusion<u32, f32>(1000, 0);
    test_inclusion<u64, f32>(1000, 0);
    test_inclusion<u32, f64>(1000, 0);
    test_inclusion<u64, f64>(1000, 0);
    test_inclusion<u32, u64>(1000, 0);
    test_inclusion<u64, u64>(1000, 0);
    test_inclusion<u32, u32>(1000, 0);
    // test_inclusion<u64, u32>(1000,0);

    test_inclusion<u32, f32>(10, 10);
    test_inclusion<u64, f32>(10, 10);
    test_inclusion<u32, f64>(10, 10);
    test_inclusion<u64, f64>(10, 10);
    test_inclusion<u32, u64>(10, 10);
    test_inclusion<u64, u64>(10, 10);
    test_inclusion<u32, u32>(10, 10);
    // test_inclusion<u64, u32>(10,10);
}

template<class morton_mode, class flt, u32 reduc_lev>
inline void test_tree(std::string dset_name) {

    using vec = sycl::vec<flt, 3>;

    f64 Nmax_flt = 1e8 * 1;

    u32 Nmax = u32(sycl::fmin(Nmax_flt, 2e9));

    auto coord_range = get_test_coord_ranges<vec>();

    auto pos
        = shamalgs::random::mock_buffer_ptr<vec>(0x111, Nmax, coord_range.lower, coord_range.upper);

    shamalgs::memory::move_buffer_on_queue(shamsys::instance::get_compute_queue(), *pos);

    std::vector<f64> times;
    std::vector<f64> Npart;

    for (f64 cnt = 1000; cnt < Nmax; cnt *= 1.1) {
        shamlog_debug_ln("TestTreePerf", cnt);
        shamsys::instance::get_compute_queue().wait();
        shambase::Timer timer;
        timer.start();

        RadixTree<morton_mode, vec> rtree = RadixTree<morton_mode, vec>(
            shamsys::instance::get_compute_queue(),
            {coord_range.lower, coord_range.upper},
            pos,
            cnt,
            reduc_lev);

        rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
        rtree.convert_bounding_box(shamsys::instance::get_compute_queue());

        shamsys::instance::get_compute_queue().wait();
        timer.end();

        times.push_back(timer.nanosec / 1.e9);
        Npart.push_back(u32(cnt));
    }

    auto &dset = shamtest::test_data().new_dataset(dset_name);

    dset.add_data("Npart", Npart);
    dset.add_data("times", times);
}

TestStart(Benchmark, "shamrock/tree/RadixTree:build:benchmark", morton_tree_build, 1) {
    test_tree<u32, f32, 0>("u32, f32, 0");
    test_tree<u64, f32, 0>("u64, f32, 0");
    test_tree<u32, f64, 0>("u32, f64, 0");
    test_tree<u64, f64, 0>("u64, f64, 0");
    test_tree<u32, f32, 1>("u32, f32, 1");
    test_tree<u64, f32, 1>("u64, f32, 1");
    test_tree<u32, f64, 1>("u32, f64, 1");
    test_tree<u64, f64, 1>("u64, f64, 1");
    test_tree<u32, f32, 2>("u32, f32, 2");
    test_tree<u64, f32, 2>("u64, f32, 2");
    test_tree<u32, f64, 2>("u32, f64, 2");
    test_tree<u64, f64, 2>("u64, f64, 2");
}

TestStart(
    Benchmark,
    "article_shamrock1:shamrock/tree/RadixTree:build:benchmark",
    morton_tree_build_article1,
    1) {
    test_tree<u32, f32, 0>("morton = u32, field type = f32");
    test_tree<u64, f32, 0>("morton = u64, field type = f32");
    test_tree<u32, f64, 0>("morton = u32, field type = f64");
    test_tree<u64, f64, 0>("morton = u64, field type = f64");
    test_tree<u32, u64, 0>("morton = u32, field type = u64");
    test_tree<u64, u64, 0>("morton = u64, field type = u64");
}
