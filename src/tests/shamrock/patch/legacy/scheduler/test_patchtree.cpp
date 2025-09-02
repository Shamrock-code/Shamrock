// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/sfc/hilbert.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamrock/scheduler/scheduler_patch_list.hpp"
#include "shamtest/shamtest.hpp"
#include <map>
#include <random>
#include <vector>

#if false

void recursprint(PatchTree &pt,std::vector<Patch>& plist,std::unordered_map<u64,u64> &idx_map, u64 idx, u32 indent){


    PatchTree::PTNode & ptnode = pt.tree[idx];


    for(u32 i = 0 ; i < indent; i++) std::cout << "   ";

    std::cout << "nodeid : " << idx << " level : " << ptnode.level << " parent id : " << ptnode.parent_id << ","<< ptnode.is_leaf <<","<< ptnode.child_are_all_leafs <<
                " -> " << ptnode.data_count << ","<< ptnode.load_value <<
                " ( " <<
                "[" << ptnode.x_min << "," << ptnode.x_max << "] " <<
                "[" << ptnode.y_min << "," << ptnode.y_max << "] " <<
                "[" << ptnode.z_min << "," << ptnode.z_max << "] " <<
                " ) ";


    if(!ptnode.is_leaf){
        std::cout << std::endl;
        for(u8 child_id = 0; child_id < 8; child_id ++){
            recursprint(pt,plist,idx_map, ptnode.childs_id[child_id], indent +1);
        }
    }else if(ptnode.linked_patchid != u64_max){
        Patch & p = plist[idx_map[ptnode.linked_patchid]];
        std::cout << " node : " <<ptnode.linked_patchid<<" ( " <<
                "[" << p.x_min << "," << p.x_max << "] " <<
                "[" << p.y_min << "," << p.y_max << "] " <<
                "[" << p.z_min << "," << p.z_max << "] " <<
                " ) " << std::endl;
    }else{
        std::cout << std::endl;
    }


}


Test_start("", testpatchtree, 1){

    std::vector<Patch> global = make_fake_patch_list(200,10);


    for(Patch & p : global){
        std::cout << p.id_patch <<

            " -> " << p.data_count << ","<< p.load_value <<
            " ( " <<
            "[" << p.x_min << "," << p.x_max << "] " <<
            "[" << p.y_min << "," << p.y_max << "] " <<
            "[" << p.z_min << "," << p.z_max << "] " <<
            " ) " << "("<<p.x_max - p.x_min<<","<<p.y_max - p.y_min<<","<<p.z_max - p.z_min<<")"

            << std::endl;
    }

    std::cout << "number of ptch : " << global.size() << std::endl;

    PatchTree pt;

    pt.build_from_patchtable(global,HilbertLB::max_box_sz);

    SchedulerPatchList plist;
    plist.global = global;

    plist.build_global_idx_map();

    pt.update_values_node(plist.global, plist.id_patch_to_global_idx);



    recursprint(pt,global,plist.id_patch_to_global_idx, 0, 0);

    std::cout << "leaf list : ";
    for(u64 idp : pt.leaf_key){
        std::cout << idp <<" , ";
    }std::cout << std::endl;

    std::cout << "parent of only leaf list : ";
    for(u64 idp : pt.parent_of_only_leaf_key){
        std::cout << idp <<" , ";
    }std::cout << std::endl;


    std::cout << "merge : ----------------------" << std::endl;


    pt.merge_node_dm1(9);
    pt.merge_node_dm1(10);
    pt.merge_node_dm1(13);
    pt.merge_node_dm1(15);

    recursprint(pt,global,plist.id_patch_to_global_idx, 0, 0);

    std::cout << "leaf list : ";
    for(u64 idp : pt.leaf_key){
        std::cout << idp <<" , ";
    }std::cout << std::endl;

    std::cout << "parent of only leaf list : ";
    for(u64 idp : pt.parent_of_only_leaf_key){
        std::cout << idp <<" , ";
    }std::cout << std::endl;


    for(auto & [key,ptnode] : pt.tree){

        if(ptnode.is_leaf){
            if(pt.leaf_key.count(key)){
                Test_assert("leaf in leaf_key set", true);
                pt.leaf_key.erase(key);
            }else{
                Test_assert("leaf in leaf_key set", false);
            }
        }

        if(ptnode.child_are_all_leafs){
            if(pt.parent_of_only_leaf_key.count(key)){
                Test_assert("parent of only leaf in parent_of_only_leaf_key set", true);
                pt.parent_of_only_leaf_key.erase(key);
            }else{
                Test_assert("parent of only leaf in parent_of_only_leaf_key set", false);
            }
        }

    }

    Test_assert("leaf key empty ", pt.leaf_key.size() == 0);
    Test_assert("parent_of_only_leaf_key empty ", pt.parent_of_only_leaf_key.size() == 0);

}

#endif
