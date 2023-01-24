#pragma once

#include "aliases.hpp"

template<class morton_t, class pos_t>
class RadixTreeMortonBuilder{public:

    static void build(
        sycl::queue & queue,
        std::tuple<pos_t,pos_t> bounding_box,
        std::unique_ptr<sycl::buffer<pos_t>> & pos_buf, 
        u32 cnt_obj, 

        std::unique_ptr<sycl::buffer<morton_t>> & out_buf_morton,
        std::unique_ptr<sycl::buffer<u32>> & out_buf_particle_index_map
    );

};