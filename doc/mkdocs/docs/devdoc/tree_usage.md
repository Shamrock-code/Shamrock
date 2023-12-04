# Shamrock Radix Tree usage

## The tree


The radix tree in shamrock can be imported by 
```c++
#include "shamrock/tree/RadixTree.hpp"
```

This define the class `RadixTree<...>`. This templated `RadixTree` can be instanciated like this : 

```c++
RadixTree<
        u32,   // The precision of the morton codes
        f32_3, // The type of the position data
        3      // The dimension 
    >
```

### Building the Tree

```c++
sycl::queue & q = shamsys::instance::get_compute_queue(); //select the queue to run on

using Tree = RadixTree<u32,f32_3,3>;

Tree rtree(
    q, //the sycl queue to build the tree on
    {coord_range.lower, coord_range.upper}, // The range of coordinates in the postions
    pos, // the position buffer
    cnt, //number of element in the position buffer
    reduc_lev //level of reduction
);

rtree.compute_cell_ibounding_box(q); //compute the sizes of the tree cells
rtree.convert_bounding_box(q); //convert the cell sizes to the original space of coordinates
```


## Tree traversal 


### Interaction Criterion 
 Here is an exemple of the definition of an interaction criterion

```c++
template<class u_morton, class flt>
class SPHTestInteractionCrit {
    using vec = sycl::vec<flt, 3>;
    public:

    // Information for the criterion
    RadixTree<u_morton, vec, 3> &tree;
    sycl::buffer<vec> &positions;
    u32 part_count;
    flt Rpart;

    // Information for the criterion that will be accessed by the GPU
    class Access {
        public:

        sycl::accessor<vec, 1, sycl::access::mode::read> part_pos;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_min;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_max;

        flt Rpart;
        flt Rpart_pow2;

        Access(SPHTestInteractionCrit crit, sycl::handler &cgh)
              : part_pos{crit.positions, cgh, sycl::read_only}, 
                Rpart(crit.Rpart),Rpart_pow2(crit.Rpart*crit.Rpart),
                tree_cell_coordrange_min{*crit.tree.buf_pos_min_cell_flt, cgh, sycl::read_only},
                tree_cell_coordrange_max{*crit.tree.buf_pos_max_cell_flt, cgh, sycl::read_only} {}

        //The values necessary to compute the interaction criterion per objects
        class ObjectValues {
            public:
            vec xyz_a;
            ObjectValues(Access acc, u32 index)
                : xyz_a(acc.part_pos[index]) {}
        };

    };

    // the interaction criterion
    inline static bool
    criterion(u32 node_index, Access acc, typename Access::ObjectValues current_values) {
        vec cur_pos_min_cell_b = acc.tree_cell_coordrange_min[node_index];
        vec cur_pos_max_cell_b = acc.tree_cell_coordrange_max[node_index];

        vec box_int_sz = {acc.Rpart,acc.Rpart,acc.Rpart};

        return 
            BBAA::cella_neigh_b(
                current_values.xyz_a - box_int_sz, current_values.xyz_a + box_int_sz, 
                cur_pos_min_cell_b, cur_pos_max_cell_b) ||
            BBAA::cella_neigh_b(
                current_values.xyz_a, current_values.xyz_a,                   
                cur_pos_min_cell_b - box_int_sz, cur_pos_min_cell_b + box_int_sz);
    };
};
```

### Traversal 


```c++
using Criterion = SPHTestInteractionCrit<morton_mode,flt>;
using CriterionAcc = typename Criterion::Access;
using CriterionVal = typename CriterionAcc::ObjectValues; 

using namespace shamrock::tree;
TreeStructureWalker walk = generate_walk<Recompute>(
    rtree.tree_struct, len_pos, SPHTestInteractionCrit<morton_mode,flt>{rtree,*pos, len_pos, rpart}
);


q.submit([&](sycl::handler &cgh) {
    auto walker        = walk.get_access(cgh);
    auto leaf_iterator = rtree.get_leaf_access(cgh);

    sycl::accessor neigh_count {neighbours,cgh,sycl::write_only, sycl::no_init};
    

    cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
        u32 sum = 0;

        CriterionVal int_values{walker.criterion(), static_cast<u32>(item.get_linear_id())};

        walker.for_each_node(
            item,int_values,
            [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                leaf_iterator.iter_object_in_leaf(
                    leaf_iterator_id, [&](u32 obj_id) { 

                        vec xyz_b = walker.criterion().part_pos[obj_id];
                        vec dxyz = xyz_b - int_values.xyz_a;
                        flt dot_ = sycl::dot(dxyz,dxyz);

                        if(dot_ < walker.criterion().Rpart_pow2){
                            sum += 1; 
                        }
                    }
                );
            },
            [&](u32 node_id) {}
        );

        neigh_count[item] = sum;
    });
    
});
```