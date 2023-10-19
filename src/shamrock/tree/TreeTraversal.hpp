// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file TreeTraversal.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamalgs/numeric.hpp"
#include <shambackends/backends/sycl/sycl.hpp>
#include "shamrock/tree/RadixTree.hpp"

namespace shamrock::tree {

    template<class u_morton, class vec>
    class ObjectIterator {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> particle_index_map;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> cell_index_map;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_min_cell;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_max_cell;

        static constexpr u32 tree_depth = RadixTree<u_morton, vec>::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        u32 leaf_offset;

        public:

        // clang-format off
        ObjectIterator(RadixTree< u_morton,  vec> & rtree,sycl::handler & cgh):
            particle_index_map{shambase::get_check_ref(rtree.tree_morton_codes.buf_particle_index_map), cgh,sycl::read_only},
            cell_index_map{shambase::get_check_ref(rtree.tree_reduced_morton_codes.buf_reduc_index_map), cgh,sycl::read_only},
            rchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_rchild_id)  , cgh,sycl::read_only},
            lchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_lchild_id)  , cgh,sycl::read_only},
            rchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_rchild_flag), cgh,sycl::read_only},
            lchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_lchild_flag), cgh,sycl::read_only},
            pos_min_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_min_cell_flt), cgh,sycl::read_only},
            pos_max_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_max_cell_flt), cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        template<class Functor_iter>
        inline void iter_object_in_cell(const u32 & cell_id, Functor_iter &&func_it) const {
            // loop on particle indexes
            uint min_ids = cell_index_map[cell_id     -leaf_offset];
            uint max_ids = cell_index_map[cell_id + 1 -leaf_offset];

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                //recover old index before morton sort
                uint id_b = particle_index_map[id_s];

                //iteration function
                func_it(id_b);

            }
            
        }

        template<class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
        inline void rtree_for(Functor_int_cd &&func_int_cd,
                                   Functor_iter &&func_it,
                                   Functor_iter_excl &&func_excl) const {
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = _nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(
                    current_node_id, pos_min_cell[current_node_id], pos_max_cell[current_node_id]);

                if (cur_id_valid) {

                    // leaf and cell can interact
                    if (current_node_id >= leaf_offset) {

                        iter_object_in_cell(current_node_id, func_it);

                        // can interact not leaf => stack
                    } else {

                        u32 lid =
                            lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                        u32 rid =
                            rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

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

        template<class Functor_int_cd, class Functor_iter>
        inline void rtree_for(Functor_int_cd &&func_int_cd, Functor_iter &&func_it) const {
            rtree_for(std::forward<Functor_int_cd>(func_int_cd),
                           std::forward<Functor_iter>(func_it),
                           [](u32) {});
        }
    };




    template<class u_morton, class vec>
    class LeafIterator {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;

        public:
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_min_cell;
        sycl::accessor<vec, 1, sycl::access::mode::read, sycl::target::device> pos_max_cell;
        private:

        static constexpr u32 tree_depth = RadixTree<u_morton, vec>::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        u32 leaf_offset;

        public:

        // clang-format off
        LeafIterator(RadixTree< u_morton,  vec> & rtree,sycl::handler & cgh):
            rchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_rchild_id)  , cgh,sycl::read_only},
            lchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_lchild_id)  , cgh,sycl::read_only},
            rchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_rchild_flag), cgh,sycl::read_only},
            lchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_lchild_flag), cgh,sycl::read_only},
            pos_min_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_min_cell_flt), cgh,sycl::read_only},
            pos_max_cell  {shambase::get_check_ref(rtree.tree_cell_ranges.buf_pos_max_cell_flt), cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        template<class Functor_int_cd, class Functor_iter, class Functor_iter_excl>
        inline void rtree_for(Functor_int_cd &&func_int_cd,
                                   Functor_iter &&func_it,
                                   Functor_iter_excl &&func_excl) const {
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = _nindex;
                stack_cursor++;

                bool cur_id_valid = func_int_cd(
                    current_node_id, pos_min_cell[current_node_id], pos_max_cell[current_node_id]);

                if (cur_id_valid) {

                    // leaf and cell can interact
                    if (current_node_id >= leaf_offset) {

                        func_it(current_node_id);

                        // can interact not leaf => stack
                    } else {

                        u32 lid =
                            lchild_id[current_node_id] + leaf_offset * lchild_flag[current_node_id];
                        u32 rid =
                            rchild_id[current_node_id] + leaf_offset * rchild_flag[current_node_id];

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

        template<class Functor_int_cd, class Functor_iter>
        inline void rtree_for(Functor_int_cd &&func_int_cd, Functor_iter &&func_it) const {
            rtree_for(std::forward<Functor_int_cd>(func_int_cd),
                           std::forward<Functor_iter>(func_it),
                           [](u32) {});
        }
    };




    template<class u_morton, class vec>
    class LeafRadixFinder {

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
        sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;

        sycl::accessor<u_morton, 1, sycl::access::mode::read, sycl::target::device> tree_morton;
        private:

        static constexpr u32 tree_depth = RadixTree<u_morton, vec>::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        u32 leaf_offset;

        public:

        // clang-format off
        LeafRadixFinder(RadixTree< u_morton,  vec> & rtree,sycl::handler & cgh):
            rchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_rchild_id)  , cgh,sycl::read_only},
            lchild_id     {shambase::get_check_ref(rtree.tree_struct.buf_lchild_id)  , cgh,sycl::read_only},
            rchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_rchild_flag), cgh,sycl::read_only},
            lchild_flag   {shambase::get_check_ref(rtree.tree_struct.buf_lchild_flag), cgh,sycl::read_only},
            tree_morton  {shambase::get_check_ref(rtree.tree_reduced_morton_codes.buf_tree_morton), cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        /**
         * @brief identify leaf owning the asked code
         * 
         * @param morton_code
         * @return u32 the leaf id owning the code in the range [0,leaf_cnt[
         */
        inline u32 identify_cell(u_morton morton_code) const {
            u32 current_node_id = 0;

            for(u32 level = 0 ; level < tree_depth; level ++) {

                u32 lid = lchild_id[current_node_id];
                u32 rid = rchild_id[current_node_id];
                u32 lflag = lchild_flag[current_node_id];
                u32 rflag = rchild_flag[current_node_id];

                u_morton m_l = tree_morton[lid];
                u_morton m_r = tree_morton[rid];

                u32 affinity_l = shambase::clz_xor(morton_code, m_l);
                u32 affinity_r = shambase::clz_xor(morton_code, m_r);

                u32 next_id = (affinity_l > affinity_r) ? lid : rid;
                u32 next_flag = (affinity_l > affinity_r) ? lflag : rflag;

                if(next_flag == 1){
                    return next_id;
                }

                current_node_id = next_id;
            }

            return u32_max;
        }

    };


    class LeafCache{public:
        sycl::buffer<u32> cnt_neigh;
        sycl::buffer<u32> scanned_cnt;
        u32 sum_neigh_cnt;
        sycl::buffer<u32> index_neigh_map;
    };

    class LeafCacheObjectIterator{

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> neigh_cnt;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> table_neigh_offset;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> table_neigh;

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> cell_owner;


        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> particle_index_map;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> cell_index_map;

        
        u32 leaf_offset;
        public:

        // clang-format off
        template<class u_morton, class vec>
        LeafCacheObjectIterator(RadixTree< u_morton,  vec> & rtree,sycl::buffer<u32> & ownerships, LeafCache & cache,sycl::handler & cgh):
            particle_index_map{shambase::get_check_ref(rtree.tree_morton_codes.buf_particle_index_map), cgh,sycl::read_only},
            cell_index_map{shambase::get_check_ref(rtree.tree_reduced_morton_codes.buf_reduc_index_map), cgh,sycl::read_only},
            neigh_cnt          {cache.cnt_neigh       ,cgh,sycl::read_only},
            table_neigh_offset {cache.scanned_cnt     ,cgh,sycl::read_only},
            table_neigh        {cache.index_neigh_map ,cgh,sycl::read_only},
            cell_owner         {ownerships            ,cgh,sycl::read_only},
            leaf_offset   (rtree.tree_struct.internal_cell_count)
        {}
        // clang-format on

        template<class Functor_iter>
        inline void iter_object_in_cell(const u32 & cell_id, Functor_iter &&func_it) const {
            // loop on particle indexes
            uint min_ids = cell_index_map[cell_id    ];
            uint max_ids = cell_index_map[cell_id + 1];

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                //recover old index before morton sort
                uint id_b = particle_index_map[id_s];

                //iteration function
                func_it(id_b);

            }
            
        }

        template<class Functor_iter>
        inline void for_each_object(u32 idx, Functor_iter &&func_it) const {

            u32 leaf_cell_owner = cell_owner[idx];
            u32 cnt = neigh_cnt[leaf_cell_owner];
            u32 offset_start = table_neigh_offset[leaf_cell_owner];
            u32 last_idx = offset_start + cnt;

            for(u32 i = offset_start; i < last_idx; i++){
                iter_object_in_cell(table_neigh[i] - leaf_offset, func_it);
            }

        }
    };


    
    struct ObjectCache;

    struct HostObjectCache{
        std::vector<u32> cnt_neigh;
        std::vector<u32> scanned_cnt;
        u32 sum_neigh_cnt;
        std::vector<u32> index_neigh_map;

        inline u64 get_memsize(){
            return (cnt_neigh.size() + scanned_cnt.size() + index_neigh_map.size() + 1)*sizeof(u32);
        }
    };

    struct ObjectCache{
        sycl::buffer<u32> cnt_neigh;
        sycl::buffer<u32> scanned_cnt;
        u32 sum_neigh_cnt;
        sycl::buffer<u32> index_neigh_map;

        inline u64 get_memsize(){
            return cnt_neigh.byte_size() + scanned_cnt.byte_size() + index_neigh_map.byte_size() + sizeof(u32);
        }

        inline HostObjectCache copy_to_host(){
            return HostObjectCache{
                shamalgs::memory::buf_to_vec(cnt_neigh, cnt_neigh.size()),
                shamalgs::memory::buf_to_vec(scanned_cnt, scanned_cnt.size()),
                sum_neigh_cnt,
                shamalgs::memory::buf_to_vec(index_neigh_map, index_neigh_map.size()),
            };
        }

        inline static ObjectCache build_from_host(HostObjectCache & cache){
            return ObjectCache{
                shamalgs::memory::vec_to_buf(cache.cnt_neigh),
                shamalgs::memory::vec_to_buf(cache.scanned_cnt),
                cache.sum_neigh_cnt,
                shamalgs::memory::vec_to_buf(cache.index_neigh_map),
            };
        }
    };

    inline ObjectCache prepare_object_cache(sycl::buffer<u32> && counts, u32 obj_cnt){


        logger::debug_sycl_ln("Cache", " reading last value ...");
        u32 neigh_last_val = shamalgs::memory::extract_element(shamsys::instance::get_compute_queue(), counts, obj_cnt-1);

        logger::debug_sycl_ln("Cache", " last value =",neigh_last_val);

        sycl::buffer<u32> neigh_scanned_vals = shamalgs::numeric::exclusive_sum(
            shamsys::instance::get_compute_queue(), 
            counts, 
            obj_cnt);

        u32 neigh_sum = neigh_last_val + shamalgs::memory::extract_element(shamsys::instance::get_compute_queue(), neigh_scanned_vals, obj_cnt-1);

        logger::debug_sycl_ln("Cache", " cache for N=",obj_cnt, "size() =",neigh_sum);

        sycl::buffer<u32> particle_neigh_map (neigh_sum);

        tree::ObjectCache pcache {
            std::move(counts),
            std::move(neigh_scanned_vals),
            neigh_sum,
            std::move(particle_neigh_map)
        };

        return pcache;
    }
    
    
    class ObjectCacheIterator{

        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> neigh_cnt;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> table_neigh_offset;
        sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> table_neigh;

        public:

        // clang-format off
        ObjectCacheIterator(ObjectCache & cache,sycl::handler & cgh):
            neigh_cnt          {cache.cnt_neigh       ,cgh,sycl::read_only},
            table_neigh_offset {cache.scanned_cnt     ,cgh,sycl::read_only},
            table_neigh        {cache.index_neigh_map ,cgh,sycl::read_only}
        {}
        // clang-format on

        template<class Functor_iter>
        inline void for_each_object(u32 idx, Functor_iter &&func_it) const {

            u32 cnt = neigh_cnt[idx];
            u32 offset_start = table_neigh_offset[idx];
            u32 last_idx = offset_start + cnt;

            for(u32 i = offset_start; i < last_idx; i++){
                func_it(table_neigh[i]);
            }

        }

        template<class Functor_iter>
        inline void for_each_object_with_id(u32 idx, Functor_iter &&func_it) const {

            u32 cnt = neigh_cnt[idx];
            u32 offset_start = table_neigh_offset[idx];
            u32 last_idx = offset_start + cnt;

            for(u32 i = offset_start; i < last_idx; i++){
                func_it(table_neigh[i],i);
            }

        }
    };




} // namespace shamrock::tree