// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once



#include "aliases.hpp"
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <set>


#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/math/vectorManip.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamrock/legacy/utils/string_utils.hpp"
#include "shamrock/sfc/morton.hpp"
#include "kernels/compute_ranges.hpp"
#include "kernels/convert_ranges.hpp"
#include "kernels/karras_alg.hpp"
#include "kernels/key_morton_sort.hpp"
#include "kernels/reduction_alg.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"



template<class T>
class RadixTreeField{
    
    RadixTreeField(
        u32 nvar, u32 cnt): 
        radix_tree_field_buf(std::make_unique<sycl::buffer<T>>(cnt*nvar)),
        nvar(nvar) {}

    
    
    public:
    u32 nvar;
    std::unique_ptr<sycl::buffer<T>> radix_tree_field_buf;

    RadixTreeField() = default;
    
    RadixTreeField(
        u32 nvar, 
        std::unique_ptr<sycl::buffer<T>> radix_tree_field_buf
        ) : nvar(nvar), radix_tree_field_buf(std::move(radix_tree_field_buf)){}

    static RadixTreeField<T> make_empty(u32 nvar, u32 cnt){
        return RadixTreeField<T>(nvar, cnt);
    }

    RadixTreeField(RadixTreeField<T> & src, sycl::buffer<u32> & id_extract_field) : 
        radix_tree_field_buf(std::make_unique<sycl::buffer<T>>(id_extract_field.size()*src.nvar)),
        nvar(src.nvar)
    {
        //cut new field according to the id map

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

            sycl::accessor acc_curr {*src.radix_tree_field_buf, cgh, sycl::read_only};
            sycl::accessor acc_other {*radix_tree_field_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc_idxs {id_extract_field, cgh, sycl::read_only};

            u32 nvar_loc = nvar;

            cgh.parallel_for(sycl::range<1>{id_extract_field.size()},  [=](sycl::item<1> i){

                const u32 gid = i.get_linear_id();
        
                const u32 idx_extr = acc_idxs[gid] * nvar_loc;
                const u32 idx_push = gid * nvar_loc;
        
                for (u32 a = 0; a < nvar_loc; a++) {
                    acc_other[idx_push + a] = acc_curr[idx_extr + a];
                }   

            });

        });
        
    }
};





template<class morton_t,class pos_t, u32 dim>
class RadixTree{

    static constexpr auto get_tree_depth = []() -> u32 {
        if constexpr (std::is_same<morton_t,u32>::value){return 32;}
        if constexpr (std::is_same<morton_t,u64>::value){return 64;}
        return 0;
    };

    RadixTree() = default;

    public:

    using ipos_t = typename shamrock::sfc::MortonCodes<morton_t, dim>::int_vec_repr;
    using coord_t = typename shamrock::math::vec_manip::VectorProperties<pos_t>::component_type;

    static constexpr u32 tree_depth = get_tree_depth();



    std::tuple<pos_t,pos_t> box_coord;

    u32 obj_cnt;
    u32 tree_leaf_count;
    u32 tree_internal_count;

    bool one_cell_mode = false;


    //build by the RadixTreeMortonBuilder 
    std::unique_ptr<sycl::buffer<morton_t>> buf_morton;
    std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;
    std::unique_ptr<sycl::buffer<u32>> buf_reduc_index_map;
    std::unique_ptr<sycl::buffer<morton_t>> buf_tree_morton; // size = leaf cnt

    //Karras alg
    std::unique_ptr<sycl::buffer<u32>>      buf_lchild_id;   // size = internal
    std::unique_ptr<sycl::buffer<u32>>      buf_rchild_id;   // size = internal
    std::unique_ptr<sycl::buffer<u8>>       buf_lchild_flag; // size = internal
    std::unique_ptr<sycl::buffer<u8>>       buf_rchild_flag; // size = internal
    std::unique_ptr<sycl::buffer<u32>>      buf_endrange;    // size = internal

    
    std::unique_ptr<sycl::buffer<ipos_t>>    buf_pos_min_cell;     // size = total count
    std::unique_ptr<sycl::buffer<ipos_t>>    buf_pos_max_cell;     // size = total count
    std::unique_ptr<sycl::buffer<pos_t>>     buf_pos_min_cell_flt; // size = total count
    std::unique_ptr<sycl::buffer<pos_t>>     buf_pos_max_cell_flt; // size = total count

    inline bool is_tree_built(){
        return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag) && bool(buf_rchild_flag) && bool(buf_endrange);
    }

    inline bool are_range_int_built(){
        return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
    }

    inline bool are_range_float_built(){
        return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
    }



    

    
    RadixTreeField<coord_t> compute_int_boxes(sycl::queue & queue,std::unique_ptr<sycl::buffer<coord_t>> & int_rad_buf, coord_t tolerance);
    
    
    void compute_cellvolume(sycl::queue & queue);


    

    



    RadixTree(
        sycl::queue & queue,
        std::tuple<pos_t,pos_t> treebox,
        std::unique_ptr<sycl::buffer<pos_t>> & pos_buf, 
        u32 cnt_obj, 
        u32 reduc_level);

    inline RadixTree(const RadixTree & other) : 
        box_coord(other.box_coord), 
        obj_cnt(other.obj_cnt), 
        tree_leaf_count(other.tree_leaf_count), 
        tree_internal_count(other.tree_internal_count),
        one_cell_mode(other.one_cell_mode),
    
        buf_morton              (syclalgs::basic::duplicate(other.buf_morton             )),
        buf_particle_index_map  (syclalgs::basic::duplicate(other.buf_particle_index_map )),
        buf_reduc_index_map     (syclalgs::basic::duplicate(other.buf_reduc_index_map    )),
        buf_tree_morton         (syclalgs::basic::duplicate(other.buf_tree_morton        )), // size = leaf cnt
        buf_lchild_id           (syclalgs::basic::duplicate(other.buf_lchild_id          )),   // size = internal
        buf_rchild_id           (syclalgs::basic::duplicate(other.buf_rchild_id          )),   // size = internal
        buf_lchild_flag         (syclalgs::basic::duplicate(other.buf_lchild_flag        )), // size = internal
        buf_rchild_flag         (syclalgs::basic::duplicate(other.buf_rchild_flag        )), // size = internal
        buf_endrange            (syclalgs::basic::duplicate(other.buf_endrange           )),    // size = internal
        buf_pos_min_cell        (syclalgs::basic::duplicate(other.buf_pos_min_cell       )),     // size = total count
        buf_pos_max_cell        (syclalgs::basic::duplicate(other.buf_pos_max_cell       )),     // size = total count
        buf_pos_min_cell_flt    (syclalgs::basic::duplicate(other.buf_pos_min_cell_flt   )), // size = total count
        buf_pos_max_cell_flt    (syclalgs::basic::duplicate(other.buf_pos_max_cell_flt   )) // size = total count
    {}

    [[nodiscard]] inline u64 memsize() const {
        u64 sum = 0;

        sum += sizeof(box_coord);
        sum += sizeof(obj_cnt);
        sum += sizeof(tree_leaf_count);
        sum += sizeof(tree_internal_count);
        sum += sizeof(one_cell_mode);

        auto add_ptr = [&](auto & a){
            if(a){
                sum += a->byte_size();
            }
        };

        add_ptr(buf_morton              );
        add_ptr(buf_particle_index_map  );
        add_ptr(buf_reduc_index_map     );
        add_ptr(buf_tree_morton         );
        add_ptr(buf_lchild_id           );
        add_ptr(buf_rchild_id           );
        add_ptr(buf_lchild_flag         );
        add_ptr(buf_rchild_flag         );
        add_ptr(buf_endrange            );
        add_ptr(buf_pos_min_cell        );
        add_ptr(buf_pos_max_cell        );
        add_ptr(buf_pos_min_cell_flt    );
        add_ptr(buf_pos_max_cell_flt    );


        return sum;
    }

    inline RadixTree duplicate(){
        const auto & cur = *this;
        return RadixTree(cur);
    }

    inline std::unique_ptr<RadixTree> duplicate_to_ptr(){
        const auto & cur = *this;
        return std::make_unique<RadixTree>(cur);
    }


    bool is_same(RadixTree & other){
        bool cmp = true;

        cmp = cmp && (one_cell_mode == other.one_cell_mode);
        cmp = cmp && (test_sycl_eq(std::get<0>(box_coord) , std::get<0>(other.box_coord)));
        cmp = cmp && (test_sycl_eq(std::get<1>(box_coord) , std::get<1>(other.box_coord)));
        cmp = cmp && (obj_cnt == other.obj_cnt);
        cmp = cmp && (tree_leaf_count == other.tree_leaf_count);
        cmp = cmp && (tree_internal_count == other.tree_internal_count);

        cmp = cmp && syclalgs::reduction::equals(*buf_morton, *other.buf_morton, obj_cnt);

        return cmp;
    }



    
    template<class T>
    using RadixTreeField = RadixTreeField<T>;

    template<class T, class LambdaComputeLeaf, class LambdaCombinator>
    RadixTreeField<T> compute_field(sycl::queue & queue,u32 nvar,
        LambdaComputeLeaf && compute_leaf, LambdaCombinator && combine) const ;


    template<class LambdaForEachCell>
    std::pair<std::set<u32>, std::set<u32>> get_walk_res_set(LambdaForEachCell && interact_cd) const;


        
    template<class LambdaForEachCell> 
    void for_each_leaf(sycl::queue & queue, LambdaForEachCell && par_for_each_cell) const;
    

    std::tuple<coord_t,coord_t> get_min_max_cell_side_lenght();


    struct CuttedTree{
        RadixTree<morton_t, pos_t, dim> rtree;
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old;

        std::unique_ptr<sycl::buffer<u32>> pdat_extract_id;
    };

    CuttedTree cut_tree(sycl::queue &queue, sycl::buffer<u8> & valid_node);

    template<class T> void print_tree_field(sycl::buffer<T> & buf_field);



    static RadixTree make_empty(){return RadixTree();}


};



template<class u_morton,class vec3, u32 dim>
template<class T, class LambdaComputeLeaf, class LambdaCombinator>
inline typename RadixTree<u_morton, vec3,dim>::template RadixTreeField<T> RadixTree<u_morton, vec3, dim>::compute_field(sycl::queue & queue,u32 nvar,

    LambdaComputeLeaf && compute_leaf, LambdaCombinator && combine) const{

    RadixTreeField<T> ret;
    ret.nvar = nvar;

    logger::debug_sycl_ln("RadixTree", "compute_field");

    ret.radix_tree_field_buf = std::make_unique<sycl::buffer<T>>(tree_internal_count + tree_leaf_count);
    sycl::range<1> range_leaf_cell{tree_leaf_count};

    queue.submit([&](sycl::handler &cgh) {
        u32 offset_leaf = tree_internal_count;

        auto tree_field = sycl::accessor{* ret.radix_tree_field_buf, cgh ,sycl::write_only, sycl::no_init};

        auto cell_particle_ids  = buf_reduc_index_map->template get_access<sycl::access::mode::read>(cgh);
        auto particle_index_map = buf_particle_index_map->template get_access<sycl::access::mode::read>(cgh);

        compute_leaf(cgh,[&](auto && lambda_loop){
            cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                u32 gid = (u32)item.get_id(0);

                u32 min_ids = cell_particle_ids[gid];
                u32 max_ids = cell_particle_ids[gid + 1];


                lambda_loop([&](auto && particle_it){
                        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
                            particle_it(particle_index_map[id_s]);
                        }
                    },
                    tree_field,
                    [&](){
                        return nvar*(offset_leaf + gid);
                    }
                );

            });
        });
        
    });

    sycl::range<1> range_tree{tree_internal_count};
    auto ker_reduc_hmax = [&](sycl::handler &cgh) {
        u32 offset_leaf = tree_internal_count;

        auto tree_field = ret.radix_tree_field_buf->template get_access<sycl::access::mode::read_write>(cgh);

        auto rchild_id   = buf_rchild_id->get_access<sycl::access::mode::read>(cgh);
        auto lchild_id   = buf_lchild_id->get_access<sycl::access::mode::read>(cgh);
        auto rchild_flag = buf_rchild_flag->get_access<sycl::access::mode::read>(cgh);
        auto lchild_flag = buf_lchild_flag->get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
            u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

            combine(
                [&](u32 nvar_id) -> T {
                    return tree_field[nvar*lid + nvar_id];
                },
                [&](u32 nvar_id) -> T {
                    return tree_field[nvar*rid + nvar_id];
                },
                tree_field,
                [&](){
                    return nvar*(gid);
                }
            );
        });
    };

    for (u32 i = 0; i < tree_depth; i++) {
        queue.submit(ker_reduc_hmax);
    }

    return std::move(ret);
}






template<class u_morton,class vec3, u32 dim>
template<class LambdaForEachCell>
inline std::pair<std::set<u32>, std::set<u32>> RadixTree<u_morton, vec3, dim>::get_walk_res_set(LambdaForEachCell && interact_cd) const {


    std::set<u32> leaf_list;
    std::set<u32> rejected_list;
    

    auto particle_index_map = sycl::host_accessor{*buf_particle_index_map};
    auto cell_index_map = sycl::host_accessor{*buf_reduc_index_map};
    auto rchild_id      = sycl::host_accessor{*buf_rchild_id  };
    auto lchild_id      = sycl::host_accessor{*buf_lchild_id  };
    auto rchild_flag    = sycl::host_accessor{*buf_rchild_flag};
    auto lchild_flag    = sycl::host_accessor{*buf_lchild_flag};

    sycl::range<1> range_leaf = sycl::range<1>{tree_leaf_count};

    u32 leaf_offset = tree_internal_count;





    u32 stack_cursor = tree_depth - 1;
    std::array<u32, tree_depth> id_stack;
    id_stack[stack_cursor] = 0;

    while (stack_cursor < tree_depth) {

        u32 current_node_id    = id_stack[stack_cursor];
        id_stack[stack_cursor] = tree_depth;
        stack_cursor++;

        


        if (interact_cd(current_node_id)) {

            // leaf and can interact => force
            if (current_node_id >= leaf_offset) {

                leaf_list.insert(current_node_id);

                // can interact not leaf => stack
            } else {

                u32 lid = lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                u32 rid = rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

                id_stack[stack_cursor - 1] = rid;
                stack_cursor--;

                id_stack[stack_cursor - 1] = lid;
                stack_cursor--;
            }
        } else {
            // grav

            rejected_list.insert(current_node_id);
        }
    }


    return std::pair<std::set<u32>, std::set<u32>>{std::move(leaf_list),std::move(rejected_list)};


}



template<class u_morton,class vec3, u32 dim>
template<class LambdaForEachCell> 
inline void RadixTree<u_morton, vec3, dim>::for_each_leaf(sycl::queue & queue, LambdaForEachCell && par_for_each_cell) const {


    queue.submit([&](sycl::handler &cgh) {

        auto particle_index_map = buf_particle_index_map-> template get_access<sycl::access::mode::read>(cgh);
        auto cell_index_map = buf_reduc_index_map-> template get_access<sycl::access::mode::read>(cgh);
        auto rchild_id      = buf_rchild_id  -> template get_access<sycl::access::mode::read>(cgh);
        auto lchild_id      = buf_lchild_id  -> template get_access<sycl::access::mode::read>(cgh);
        auto rchild_flag    = buf_rchild_flag-> template get_access<sycl::access::mode::read>(cgh);
        auto lchild_flag    = buf_lchild_flag-> template get_access<sycl::access::mode::read>(cgh);

        sycl::range<1> range_leaf = sycl::range<1>{tree_leaf_count};

        u32 leaf_offset = tree_internal_count;

        
        

        auto par_for = [&](auto && for_each_leaf){

            cgh.parallel_for(range_leaf, [=](sycl::item<1> item) {

                u32 id_cell_a = (u32)item.get_id(0) + leaf_offset;

                auto iter_obj_cell = [&](u32 cell_id, auto && func_it){
                    uint min_ids = cell_index_map[cell_id     -leaf_offset];
                    uint max_ids = cell_index_map[cell_id + 1 -leaf_offset];

                    for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                        //recover old index before morton sort
                        uint id_b = particle_index_map[id_s];

                        //iteration function
                        func_it(id_b);

                    }
                };

                auto walk_loop = [&](u32 id_cell_a, auto && for_other_cell){

                    u32 stack_cursor = tree_depth - 1;
                    std::array<u32, tree_depth> id_stack;
                    id_stack[stack_cursor] = 0;

                    while (stack_cursor < tree_depth) {

                        u32 current_node_id    = id_stack[stack_cursor];
                        id_stack[stack_cursor] = tree_depth;
                        stack_cursor++;

                        

                        auto walk_logic = [&](const bool & cur_id_valid , auto && func_leaf_found, auto && func_node_rejected){

                            if (cur_id_valid) {

                                // leaf and can interact => force
                                if (current_node_id >= leaf_offset) {

                                    func_leaf_found();

                                    // can interact not leaf => stack
                                } else {

                                    u32 lid = lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                                    u32 rid = rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

                                    id_stack[stack_cursor - 1] = rid;
                                    stack_cursor--;

                                    id_stack[stack_cursor - 1] = lid;
                                    stack_cursor--;
                                }
                            } else {
                                // grav

                                func_node_rejected();
                            }
                        };

                        for_other_cell(current_node_id,walk_logic);

                    }

                };

                for_each_leaf(id_cell_a, walk_loop,iter_obj_cell);

            });


        };

        par_for_each_cell(cgh,par_for);

    });
}


template<class u_morton,class vec3, u32 dim>
inline auto RadixTree<u_morton, vec3, dim>::get_min_max_cell_side_lenght() -> std::tuple<coord_t,coord_t>{

    u32 len = tree_leaf_count;

    sycl::buffer<coord_t> min_side_lenght {len};
    sycl::buffer<coord_t> max_side_lenght {len};

    auto & q = shamsys::instance::get_compute_queue();

    q.submit([&](sycl::handler &cgh) {
        u32 offset_leaf = tree_internal_count;

        sycl::accessor pos_min_cell { *buf_pos_min_cell_flt,cgh,sycl::read_only};
        sycl::accessor pos_max_cell { *buf_pos_max_cell_flt,cgh,sycl::read_only};

        sycl::accessor s_lengh_min { min_side_lenght,cgh,sycl::write_only,sycl::no_init};
        sycl::accessor s_lengh_max { max_side_lenght,cgh,sycl::write_only,sycl::no_init};

        sycl::range<1> range_tree{tree_leaf_count};

        cgh.parallel_for(range_tree, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            vec3 min = pos_min_cell[gid + offset_leaf];
            vec3 max = pos_max_cell[gid + offset_leaf];

            vec3 sz = max - min;

            s_lengh_min[gid] = sycl::fmin(sycl::fmin(sz.x(),sz.y()),sz.z());
            s_lengh_max[gid] = sycl::fmax(sycl::fmax(sz.x(),sz.y()),sz.z());
        });
    });

    

    coord_t min = syclalgs::reduction::reduce(q, min_side_lenght, 0,len,sycl::minimum<coord_t>{});
    coord_t max = syclalgs::reduction::reduce(q, max_side_lenght, 0,len,sycl::maximum<coord_t>{});

    return {min,max};
}





namespace tree_comm {

    

    template<class u_morton,class vec3>
    class RadixTreeMPIRequest{public:
        template<class T> using Request = mpi_sycl_interop::BufferMpiRequest<T>;
        using RTree = RadixTree<u_morton,vec3,3>;

        mpi_sycl_interop::comm_type comm_mode;
        mpi_sycl_interop::op_type comm_op;

        RTree &rtree;

        std::vector<Request<u_morton>> rq_u_morton;
        std::vector<Request<u32>> rq_u32;
        std::vector<Request<u8>> rq_u8;
        std::vector<Request<vec3>> rq_vec;


        std::vector<Request<typename RTree::ipos_t>> rq_vec3i;

        inline RadixTreeMPIRequest(
            RTree & rtree,
            mpi_sycl_interop::op_type comm_op
            ) : rtree(rtree) , comm_mode(mpi_sycl_interop::current_mode) , comm_op(comm_op) {}

        inline void finalize(){
            mpi_sycl_interop::waitall(rq_u_morton);
            mpi_sycl_interop::waitall(rq_u32);
            mpi_sycl_interop::waitall(rq_u8);
            mpi_sycl_interop::waitall(rq_vec3i);
            mpi_sycl_interop::waitall(rq_vec);

            if(comm_op == mpi_sycl_interop::Recv_Probe){
                rtree.obj_cnt = rtree.buf_morton->size();
                rtree.tree_leaf_count = rtree.buf_tree_morton->size();
                rtree.tree_internal_count = rtree.buf_lchild_id->size();

                {
                    sycl::host_accessor bmin {*rtree.buf_pos_min_cell_flt};
                    sycl::host_accessor bmax {*rtree.buf_pos_max_cell_flt};

                    rtree.box_coord = {bmin[0],bmax[0]};
                }

                //One cell mode check

                {
                    sycl::host_accessor indmap {*rtree.buf_reduc_index_map};
                    rtree.one_cell_mode = (indmap[rtree.buf_reduc_index_map->size()-1] == 0);
                }

            }
        }

    };

    template<class u_morton,class vec3>
    inline void wait_all(std::vector<RadixTreeMPIRequest<u_morton,vec3>> & rqs){
        for (auto & rq : rqs) {
            rq.finalize();
        }
    }





    template<class u_morton,class vec3>
    inline u64 comm_isend(RadixTree<u_morton,vec3,3> &rtree, std::vector<RadixTreeMPIRequest<u_morton,vec3>> & rqs,i32 rank_dest, 
            i32 tag,
            MPI_Comm comm){

        u64 ret_len = 0;

        rqs.push_back(RadixTreeMPIRequest<u_morton,vec3>(
            rtree,
            mpi_sycl_interop::op_type::Send
        ));

        auto & rq = rqs.back();
        


        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_morton,rq.rtree.obj_cnt, rq.rq_u_morton, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_particle_index_map,rq.rtree.obj_cnt, rq.rq_u32, rank_dest, tag, comm);

        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_reduc_index_map, rq.rtree.tree_leaf_count+1, rq.rq_u32, rank_dest, tag, comm);

        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_tree_morton, rq.rtree.tree_leaf_count,rq.rq_u_morton, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_lchild_id,rq.rtree.tree_internal_count, rq.rq_u32, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_rchild_id,rq.rtree.tree_internal_count, rq.rq_u32, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_lchild_flag,rq.rtree.tree_internal_count, rq.rq_u8, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_rchild_flag,rq.rtree.tree_internal_count, rq.rq_u8, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_endrange,rq.rtree.tree_internal_count, rq.rq_u32, rank_dest, tag, comm);

        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_pos_min_cell,rq.rtree.tree_internal_count + rq.rtree.tree_leaf_count, rq.rq_vec3i, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_pos_max_cell,rq.rtree.tree_internal_count + rq.rtree.tree_leaf_count, rq.rq_vec3i, rank_dest, tag, comm);

        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_pos_min_cell_flt,rq.rtree.tree_internal_count + rq.rtree.tree_leaf_count, rq.rq_vec, rank_dest, tag, comm);
        ret_len += mpi_sycl_interop::isend(rq.rtree.buf_pos_max_cell_flt,rq.rtree.tree_internal_count + rq.rtree.tree_leaf_count, rq.rq_vec, rank_dest, tag, comm);

        return ret_len;
    }
    


    template<class u_morton,class vec3>
    inline u64 comm_irecv_probe(RadixTree<u_morton,vec3,3> &rtree, std::vector<RadixTreeMPIRequest<u_morton,vec3>> & rqs,i32 rank_source, 
            i32 tag,
            MPI_Comm comm){

        rqs.push_back(RadixTreeMPIRequest<u_morton,vec3>(
            rtree,
            mpi_sycl_interop::op_type::Recv_Probe
        ));

        auto & rq = rqs.back();
        
        u64 ret_len = 0;

        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_morton, rq.rq_u_morton, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_particle_index_map, rq.rq_u32, rank_source, tag, comm);

        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_reduc_index_map, rq.rq_u32, rank_source, tag, comm);

        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_tree_morton,rq.rq_u_morton, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_lchild_id, rq.rq_u32, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_rchild_id, rq.rq_u32, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_lchild_flag, rq.rq_u8, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_rchild_flag, rq.rq_u8, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_endrange, rq.rq_u32, rank_source, tag, comm);

        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_pos_min_cell, rq.rq_vec3i, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_pos_max_cell, rq.rq_vec3i, rank_source, tag, comm);


        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_pos_min_cell_flt, rq.rq_vec, rank_source, tag, comm);
        ret_len += mpi_sycl_interop::irecv_probe(rq.rtree.buf_pos_max_cell_flt, rq.rq_vec, rank_source, tag, comm);

        return ret_len;
    }

}


//TODO move h iter thing + multipoles to a tree field class




namespace walker {

    namespace interaction_crit {
        template<class vec3,class flt>
        inline bool sph_radix_cell_crit(vec3 xyz_a,vec3 part_a_box_min,vec3 part_a_box_max,vec3 cur_cell_box_min,vec3 cur_cell_box_max,flt box_int_sz){
            
            vec3 inter_box_b_min = cur_cell_box_min - box_int_sz;
            vec3 inter_box_b_max = cur_cell_box_max + box_int_sz;

            return 
                BBAA::cella_neigh_b(
                    part_a_box_min, part_a_box_max, 
                    cur_cell_box_min, cur_cell_box_max) ||
                BBAA::cella_neigh_b(
                    xyz_a, xyz_a,                   
                    inter_box_b_min, inter_box_b_max);
        }


        template<class vec3,class flt>
        inline bool sph_cell_cell_crit(vec3 cella_min,vec3 cella_max,vec3 cellb_min, vec3 cellb_max, flt rint_a, flt rint_b){

            vec3 inter_box_a_min = cella_min - rint_a;
            vec3 inter_box_a_max = cella_max + rint_a;

            vec3 inter_box_b_min = cellb_min - rint_b;
            vec3 inter_box_b_max = cellb_max + rint_b;

            return BBAA::cella_neigh_b(inter_box_a_min, inter_box_a_max, cellb_min,cellb_max) ||
                BBAA::cella_neigh_b(inter_box_b_min, inter_box_b_max, cella_min,cella_max) ;

        }
    }


    

    template<class u_morton,class vec3>
    class Radix_tree_accessor{public:
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  particle_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device> cell_index_map;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  rchild_id     ;
        sycl::accessor<u32 ,1,sycl::access::mode::read,sycl::target::device>  lchild_id     ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  rchild_flag   ;
        sycl::accessor<u8  ,1,sycl::access::mode::read,sycl::target::device>  lchild_flag   ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_min_cell  ;
        sycl::accessor<vec3,1,sycl::access::mode::read,sycl::target::device>  pos_max_cell  ;

        static constexpr u32 tree_depth = RadixTree<u_morton,vec3,3>::tree_depth;
        static constexpr u32 _nindex = 4294967295;

        u32 leaf_offset;

        
        Radix_tree_accessor(RadixTree< u_morton,  vec3,3> & rtree,sycl::handler & cgh):
            particle_index_map(rtree.buf_particle_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            cell_index_map(rtree.buf_reduc_index_map-> template get_access<sycl::access::mode::read>(cgh)),
            rchild_id     (rtree.buf_rchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            lchild_id     (rtree.buf_lchild_id  -> template get_access<sycl::access::mode::read>(cgh)),
            rchild_flag   (rtree.buf_rchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            lchild_flag   (rtree.buf_lchild_flag-> template get_access<sycl::access::mode::read>(cgh)),
            pos_min_cell  (rtree.buf_pos_min_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            pos_max_cell  (rtree.buf_pos_max_cell_flt-> template get_access<sycl::access::mode::read>(cgh)),
            leaf_offset   (rtree.tree_internal_count)
        {}
    };


    template<class Rta,class Functor_iter>
    inline void iter_object_in_cell(const Rta &acc,const u32 & cell_id, Functor_iter &&func_it){
        // loop on particle indexes
        uint min_ids = acc.cell_index_map[cell_id     -acc.leaf_offset];
        uint max_ids = acc.cell_index_map[cell_id + 1 -acc.leaf_offset];

        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

            //recover old index before morton sort
            uint id_b = acc.particle_index_map[id_s];

            //iteration function
            func_it(id_b);

        }
        

        /*
        std::array<u32, 16> stack_run;

        u32 run_cursor = 16;

        auto is_stack_full = [&]() -> bool{
            return run_cursor == 0;
        };

        auto is_stack_not_empty = [&]() -> bool{
            return run_cursor < 16;
        };

        auto push_stack = [&](u32 val){
            run_cursor --;
            stack_run[run_cursor] = val;
        };

        auto pop_stack = [&]() -> u32 {
            u32 v = stack_run[run_cursor];
            run_cursor ++;
            return v;
        };

        auto empty_stack = [&](){
            while (is_stack_not_empty()) {
                func_it(pop_stack());
            }
        };

        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
            uint id_b = acc.particle_index_map[id_s];

            if(is_stack_full()){
                empty_stack();
            }
            
            push_stack(id_b);

        }

        empty_stack();
        */
        
    }


    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
    inline void rtree_for_cell(const Rta &acc, Functor_int_cd &&func_int_cd, Functor_iter &&func_it, Functor_iter_excl &&func_excl) {
        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        while (stack_cursor < Rta::tree_depth) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and cell can interact
                if (current_node_id >= acc.leaf_offset) {

                    func_it(current_node_id);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                func_excl(current_node_id);
            }
        }
    }


    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
    inline void rtree_for(const Rta &acc, Functor_int_cd &&func_int_cd, Functor_iter &&func_it, Functor_iter_excl &&func_excl) {
        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        while (stack_cursor < Rta::tree_depth) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and can interact => force
                if (current_node_id >= acc.leaf_offset) {

                    // loop on particle indexes
                    //uint min_ids = acc.cell_index_map[current_node_id     -acc.leaf_offset];
                    //uint max_ids = acc.cell_index_map[current_node_id + 1 -acc.leaf_offset];
                    //
                    //for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {
                    //
                    //    //recover old index before morton sort
                    //    uint id_b = acc.particle_index_map[id_s];
                    //
                    //    //iteration function
                    //    func_it(id_b);
                    //}

                    iter_object_in_cell(acc, current_node_id, func_it);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                func_excl(current_node_id);
            }
        }
    }

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl, class arr_type>
    inline void rtree_for_fill_cache(Rta &acc,arr_type & cell_cache, Functor_int_cd &&func_int_cd) {

        constexpr u32 cache_sz = cell_cache.size();
        u32 cache_pos = 0;

        auto push_in_cache = [&cell_cache,&cache_pos](u32 id){
            cell_cache[cache_pos] = id;
            cache_pos ++;
        };

        u32 stack_cursor = Rta::tree_depth - 1;
        std::array<u32, Rta::tree_depth> id_stack;
        id_stack[stack_cursor] = 0;

        auto get_el_cnt_in_stack = [&]() -> u32{
            return Rta::tree_depth - stack_cursor;
        };

        while ((stack_cursor < Rta::tree_depth) && (cache_pos + get_el_cnt_in_stack < cache_sz)) {

            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;

            bool cur_id_valid = func_int_cd(current_node_id);

            if (cur_id_valid) {

                // leaf and can interact => force
                if (current_node_id >= acc.leaf_offset) {

                    //can interact => add to cache
                    push_in_cache(current_node_id);

                    // can interact not leaf => stack
                } else {

                    u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                    u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // grav
                //.....
            }
        }

        while (stack_cursor < Rta::tree_depth) {
            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = Rta::_nindex;
            stack_cursor++;
            push_in_cache(current_node_id);
        }

        if(cache_pos < cache_sz){
            push_in_cache(u32_max);
        }
    }

    template <class Rta, class Functor_int_cd, class Functor_iter, class Functor_iter_excl, class arr_type>
    inline void rtree_for(Rta &acc,arr_type & cell_cache, Functor_int_cd &&func_int_cd, Functor_iter &&func_it) {

        constexpr u32 cache_sz = cell_cache.size();

        std::array<u32, Rta::tree_depth> id_stack;

        auto walk_step = [&](u32 start_id){
            u32 stack_cursor = Rta::tree_depth - 1;
            id_stack[stack_cursor] = start_id;

            while (stack_cursor < Rta::tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = Rta::_nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(current_node_id);

                if (cur_id_valid) {

                    // leaf and can interact => force
                    if (current_node_id >= acc.leaf_offset) {

                        // loop on particle indexes
                        uint min_ids = acc.cell_index_map[current_node_id     -acc.leaf_offset];
                        uint max_ids = acc.cell_index_map[current_node_id + 1 -acc.leaf_offset];

                        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                            //recover old index before morton sort
                            uint id_b = acc.particle_index_map[id_s];

                            //iteration function
                            func_it(id_b);
                        }

                        // can interact not leaf => stack
                    } else {

                        u32 lid = acc.lchild_id[current_node_id] + acc.leaf_offset * acc.lchild_flag[current_node_id];
                        u32 rid = acc.rchild_id[current_node_id] + acc.leaf_offset * acc.rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                } else {
                    // grav
                    //...
                }
            }
        };

        for (u32 cache_pos = 0; cache_pos < cache_sz && cell_cache[cache_pos] != u32_max; cache_pos ++) {
            walk_step(cache_pos);
        }

        
    }

    



}