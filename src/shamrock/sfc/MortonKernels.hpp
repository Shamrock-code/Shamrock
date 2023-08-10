// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/sfc/morton.hpp"




namespace shamrock::sfc{

    namespace details {


        template<class morton_t>
        struct MortonInfo{
            static constexpr morton_t err_code = 0x0;
        };

        template<>
        struct MortonInfo<u32>{
            static constexpr u32 err_code = 4294967295U;
        };

        template<>
        struct MortonInfo<u64>{
            static constexpr u64 err_code = 18446744073709551615UL;
        };

        /**
        * @brief fill the end of a buffer (indices from morton_count up to fill_count-1) with error values (maximum int value)
        * 
        * @param queue sycl queue
        * @param morton_count lenght of the morton buffer 
        * @param fill_count final lenght to be filled with error value
        * @param buf_morton morton buffer that will be updated
        */
        template<class morton_t>
        void sycl_fill_trailling_buffer(
            sycl::queue & queue,
            u32 morton_count,
            u32 fill_count,
            std::unique_ptr<sycl::buffer<morton_t>> & buf_morton
            );
    } // namespace details

    template<class morton_t,class _pos_t, u32 dim>
    class MortonKernels{

        public:

        
        using Morton = MortonCodes<morton_t, dim>;

        using pos_t = _pos_t;
        using coord_t = typename shambase::VectorProperties<pos_t>::component_type;
        using ipos_t  = typename Morton::int_vec_repr;
        using int_t = typename Morton::int_vec_repr_base;

        using CoordTransform = shammath::CoordRangeTransform< typename Morton::int_vec_repr,_pos_t>;

        inline static CoordTransform get_transform(pos_t bounding_box_min,pos_t bounding_box_max){
            return MortonConverter<morton_t, pos_t, dim>::get_transform(bounding_box_min, bounding_box_max);
        }
        inline static ipos_t to_morton_grid(pos_t pos, CoordTransform transform){
            return MortonConverter<morton_t, pos_t, dim>::to_morton_grid(pos, transform);
        }


        inline static pos_t to_real_space(ipos_t pos, CoordTransform transform){
            return MortonConverter<morton_t, pos_t, dim>::to_real_space(pos, transform);
        }

        /**
        * @brief convert a buffer of 3d positions to morton codes
        * 
        * @param queue sycl queue
        * @param pos_count lenght of the position buffer 
        * @param in_positions 
        * @param bounding_box_min 
        * @param bounding_box_max 
        * @param out_morton 
        */
        static void sycl_xyz_to_morton(
            sycl::queue & queue,
            u32 pos_count,
            sycl::buffer<pos_t> & in_positions,
            pos_t bounding_box_min,
            pos_t bounding_box_max,
            std::unique_ptr<sycl::buffer<morton_t>> & out_morton);

        /**
        * @brief fill the end of a buffer (indices from morton_count up to fill_count-1) with error values (maximum int value)
        * 
        * @param queue sycl queue
        * @param morton_count lenght of the morton buffer 
        * @param fill_count final lenght to be filled with error value
        * @param buf_morton morton buffer that will be updated
        */
        inline static void sycl_fill_trailling_buffer(
            sycl::queue & queue,
            u32 morton_count,
            u32 fill_count,
            std::unique_ptr<sycl::buffer<morton_t>> & buf_morton
            ){
                details::sycl_fill_trailling_buffer<morton_t>(queue, morton_count, fill_count, buf_morton);
        }


        static void sycl_irange_to_range(sycl::queue & queue,
            u32 buf_len , 
            pos_t bounding_box_min,
            pos_t bounding_box_max,
            std::unique_ptr<sycl::buffer<ipos_t>> & buf_pos_min_cell,
            std::unique_ptr<sycl::buffer<ipos_t>> & buf_pos_max_cell,
            std::unique_ptr<sycl::buffer<pos_t>> & out_buf_pos_min_cell_flt,
            std::unique_ptr<sycl::buffer<pos_t>> & out_buf_pos_max_cell_flt);

    };

} // namespace shamrock::sfc