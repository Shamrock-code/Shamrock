# SchedulerMPI

The MPI Scheduler is the class that handle the distribution of Patches / Data on the cluster.

## Scheduler step


```{prf:algorithm} mpi scheduler step
1. Initialisation
    &ensp;&thinsp; update patch list
    &ensp;&thinsp; rebuild patch index map
    &ensp;&thinsp; apply 2 reduction step patchtree
    &ensp;&thinsp; Generate merge and split request

2. Patch Splitting
    &ensp;&thinsp; apply split requests
    &ensp;&thinsp; update ```PatchTree```

3. Patch Merging & LB
    &ensp;&thinsp; update packing index
    &ensp;&thinsp; update patch list
    &ensp;&thinsp; generate LB change list
    &ensp;&thinsp; apply LB change list
    &ensp;&thinsp; apply merge requests
    &ensp;&thinsp; update ```PatchTree```
    &ensp;&thinsp; if(Merge) update patch list

4. ```PatchTree``` reduce of sub fields
5. rebuild local table
```

Implementation :

```cpp

void scheduler_step(bool do_split_merge,bool do_load_balancing){

    // update patch list
    patch_list.sync_global();


    if(do_split_merge){
        // rebuild patch index map
        patch_list.build_global_idx_map();

        // apply reduction on leafs and corresponding parents
        patch_tree.partial_values_reduction(
            patch_list.global,
            patch_list.id_patch_to_global_idx);

        // Generate merge and split request
        std::unordered_set<u64> split_rq = patch_tree.get_split_request(crit_patch_split);
        std::unordered_set<u64> merge_rq = patch_tree.get_merge_request(crit_patch_merge);


        // apply split requests
        // update patch_list.global same on every node
        // and split patchdata accordingly if owned
        // & update tree
        split_patches(split_rq);

        // update packing index
        // same operation on evey nodes
        set_patch_pack_values(merge_rq);

        // update patch list
        // necessary to update load values in splitted patches
        // alternative : disable this step and set fake load values (load parent / 8)
        //alternative impossible if gravity because we have to compute the multipole
        //owned_patch_id = patch_list.build_local();
        //patch_list.sync_global();
        // not necessary we use fake values and compute the real ones latter
    }

    if(do_load_balancing){
        // generate LB change list
        std::vector<std::tuple<u64, i32, i32,i32>> change_list =
            make_change_list(patch_list.global);

        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
    }

    if(do_split_merge){
        // apply merge requests
        // & update tree
        merge_patches(merge_rq);


        // if(Merge) update patch list
        if(! merge_rq.empty()){
            owned_patch_id = patch_list.build_local();
            patch_list.sync_global();
        }
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}

```


## split_patches


```{prf:algorithm} scheduler patch splitter
for all patch to split :
    &ensp;&thinsp; Split patch in PatchTree
    &ensp;&thinsp; Split patch in PatchList
    &ensp;&thinsp; Split patch in patchData
```

```cpp
void split_patches(std::unordered_set<u64> split_rq){
    for(u64 tree_id : split_rq){

        patch_tree.split_node(tree_id);
        PTNode & splitted_node = patch_tree.tree[tree_id];

        auto [idx_p0,idx_p1,idx_p2,idx_p3,idx_p4,idx_p5,idx_p6,idx_p7]
            =  patch_list.split_patch(splitted_node.linked_patchid);

        u64 old_patch_id = splitted_node.linked_patchid;

        splitted_node.linked_patchid = u64_max;
        patch_tree.tree[splitted_node.childs_id[0]].linked_patchid = patch_list.global[idx_p0].id_patch;
        patch_tree.tree[splitted_node.childs_id[1]].linked_patchid = patch_list.global[idx_p1].id_patch;
        patch_tree.tree[splitted_node.childs_id[2]].linked_patchid = patch_list.global[idx_p2].id_patch;
        patch_tree.tree[splitted_node.childs_id[3]].linked_patchid = patch_list.global[idx_p3].id_patch;
        patch_tree.tree[splitted_node.childs_id[4]].linked_patchid = patch_list.global[idx_p4].id_patch;
        patch_tree.tree[splitted_node.childs_id[5]].linked_patchid = patch_list.global[idx_p5].id_patch;
        patch_tree.tree[splitted_node.childs_id[6]].linked_patchid = patch_list.global[idx_p6].id_patch;
        patch_tree.tree[splitted_node.childs_id[7]].linked_patchid = patch_list.global[idx_p7].id_patch;

        patch_data.split_patchdata(
            old_patch_id,
            patch_list.global[idx_p0],
            patch_list.global[idx_p1],
            patch_list.global[idx_p2],
            patch_list.global[idx_p3],
            patch_list.global[idx_p4],
            patch_list.global[idx_p5],
            patch_list.global[idx_p6],
            patch_list.global[idx_p7]);

    }
}
```




## merge_patches


```{prf:algorithm} scheduler patch splitter
for all patch to split :
    &ensp;&thinsp; Get ids of patches to merge
    &ensp;&thinsp; merge patchdata(s)
    &ensp;&thinsp; merge patch object in SchedulerPatchList
    &ensp;&thinsp; merge corresponding node in patchtree
```

```cpp
void merge_patches(std::unordered_set<u64> merge_rq){
    for(u64 tree_id : merge_rq){

        PTNode & to_merge_node = patch_tree.tree[tree_id];

        std::cout << "merging patch tree id : " << tree_id << "\n";


        u64 patch_id0 = patch_tree.tree[to_merge_node.childs_id[0]].linked_patchid;
        u64 patch_id1 = patch_tree.tree[to_merge_node.childs_id[1]].linked_patchid;
        u64 patch_id2 = patch_tree.tree[to_merge_node.childs_id[2]].linked_patchid;
        u64 patch_id3 = patch_tree.tree[to_merge_node.childs_id[3]].linked_patchid;
        u64 patch_id4 = patch_tree.tree[to_merge_node.childs_id[4]].linked_patchid;
        u64 patch_id5 = patch_tree.tree[to_merge_node.childs_id[5]].linked_patchid;
        u64 patch_id6 = patch_tree.tree[to_merge_node.childs_id[6]].linked_patchid;
        u64 patch_id7 = patch_tree.tree[to_merge_node.childs_id[7]].linked_patchid;


        std::cout << format("  -> (%d %d %d %d %d %d %d %d)\n", patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);

        if(patch_list.global[patch_list.id_patch_to_global_idx[ patch_id0 ]].node_owner_id == shamcomm::world_rank()){
            patch_data.merge_patchdata(patch_id0, patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);
        }

        patch_list.merge_patch(
            patch_list.id_patch_to_global_idx[ patch_id0 ],
            patch_list.id_patch_to_global_idx[ patch_id1 ],
            patch_list.id_patch_to_global_idx[ patch_id2 ],
            patch_list.id_patch_to_global_idx[ patch_id3 ],
            patch_list.id_patch_to_global_idx[ patch_id4 ],
            patch_list.id_patch_to_global_idx[ patch_id5 ],
            patch_list.id_patch_to_global_idx[ patch_id6 ],
            patch_list.id_patch_to_global_idx[ patch_id7 ]);

        patch_tree.merge_node_dm1(tree_id);

        to_merge_node.linked_patchid = patch_id0;

    }
}
```


# New implementation :

layout setting :

```cpp
patchdata_layout::reset_fields();

patchdata_layout::add_field(FieldDescriptor<f32>("rho",1));
patchdata_layout::add_field(FieldDescriptor<f32_3>("axyz",1));

patchdata_layout::add_field(FieldDescriptor<f32>("dustfrac",nbins));
patchdata_layout::add_field(FieldDescriptor<f32_3>("dustdeltv",nbins));

patchdata_layout::commit_layout();

SchedulerBuilder buildsched();

DataBuilder dbuild = setup_fcc ......

buildsched.add_data(dbuild)

PatchScheduler psched = dbuild.build(1e6, 1e6/8);

run_sim();

```
