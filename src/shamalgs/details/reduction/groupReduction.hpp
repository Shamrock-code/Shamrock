// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/memory.hpp"
#include "shambase/sycl.hpp"
#include "shambase/sycl_utils.hpp"

template<class T,u32 work_group_size>
class KernelSliceReduceSum;

namespace shamalgs::reduction::details {

    template<class T,u32 work_group_size>
    struct GroupReduction{

        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    };


    template<class T,u32 work_group_size>
    inline T GroupReduction<T, work_group_size>::sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id){
        u32 len = end_id - start_id;

            sycl::buffer<T> buf_int(len);

            shamalgs::memory::write_with_offset_into(buf_int, buf1, start_id, len);

            u32 cur_slice_sz = 1;
            u32 remaining_val = len;
            while(len / cur_slice_sz > work_group_size*8){
                
                sycl::nd_range<1> exec_range = shambase::make_range(remaining_val,work_group_size);
                
                q.submit([&](sycl::handler &cgh) {

                    sycl::accessor global_mem{buf_int, cgh, sycl::read_write};

                    u32 slice_read_size = cur_slice_sz;
                    u32 slice_write_size = cur_slice_sz * work_group_size;
                    u32 max_id = len;

                    cgh.parallel_for<KernelSliceReduceSum<T,work_group_size>>(
                        exec_range,
                        [=](sycl::nd_item<1> item) {

                        u64 lid = item.get_local_id(0);
                        u64 group_tile_id = item.get_group_linear_id();
                        u64 gid = group_tile_id * work_group_size + lid;

                        u64 iread = gid*slice_read_size;
                        u64 iwrite = group_tile_id*slice_write_size;

                        T val_read = (iread < max_id) ? global_mem[iread] : T{0};

                        #ifdef SYCL_COMP_DPCPP
                        T local_red = sycl::reduce_over_group(item.get_group(), val_read, sycl::plus<>{});
                        #endif

                        #ifdef SYCL_COMP_OPENSYCL
                        T local_red = sycl::reduce_over_group(item.get_group(), val_read, sycl::plus<T>{});
                        #endif

                        #ifdef SYCL_COMP_SYCLUNKNOWN
                        T local_red = sycl::reduce_over_group(item.get_group(), val_read, sycl::plus<T>{});
                        #endif

                        //can be removed if i change the index in the look back ?
                        if(lid == 0){
                            global_mem[iwrite] = local_red;
                        }

                    });
                });

                cur_slice_sz *= work_group_size;
                remaining_val = exec_range.get_group_range().size();
            }

            sycl::buffer<T> recov {remaining_val};

            sycl::nd_range<1> exec_range = shambase::make_range(remaining_val,work_group_size);
            q.submit([&,remaining_val](sycl::handler &cgh) {

                sycl::accessor compute_buf{buf_int, cgh, sycl::read_only};
                sycl::accessor result{recov, cgh, sycl::write_only, sycl::no_init};

                u32 slice_read_size = cur_slice_sz;

                cgh.parallel_for(
                    exec_range,
                    [=](sycl::nd_item<1> item) {

                    u64 lid = item.get_local_id(0);
                    u64 group_tile_id = item.get_group_linear_id();
                    u64 gid = group_tile_id * work_group_size + lid;

                    u64 iread = gid*slice_read_size;

                    if(gid >= remaining_val){
                        return;
                    }

                    result[gid] = compute_buf[iread];
                    

                });
            });


            T ret {0};
            {
                sycl::host_accessor acc {recov, sycl::read_only};
                for(u64 i = 0; i < remaining_val; i++){
                    ret += acc[i];
                }
            }

            return ret;
    }

} // namespace shamalgs::reduction::details