// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamcomm/mpi.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

inline void check_buf(std::string prefix, sycl::buffer<u8> & b1, sycl::buffer<u8> & b2){


    shamtest::asserts().assert_equal(prefix+std::string("same size"), b1.size(), b2.size());

    {
        sycl::host_accessor acc1 {b1};
        sycl::host_accessor acc2 {b2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for(u32 i = 0; i < b1.size(); i++){
            if(!shambase::vec_equals(acc1[i] , acc2[i])){
                eq = false;
                //id_err_list += std::to_string(i) + " ";
            }
        }

        if (eq) {
            shamtest::asserts().assert_bool("same content", eq);
        }else{
            shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
        }
    }
}

TestStart(Unittest, "shamsys/comm/CommunicationBuffer/constructor", test_basic_serialized_constr, 1){


    u32 nbytes = 1e5;
    sycl::buffer<u8> buf_comp = shamalgs::random::mock_buffer<u8>(0x111, nbytes);

    {
        shamcomm::CommunicationBuffer cbuf {buf_comp, sham::get_queue_details()};
        sycl::buffer<u8> ret = cbuf.copy_back();
        check_buf("copy to host mode", buf_comp, ret);
    }

    {
        shamcomm::CommunicationBuffer cbuf {buf_comp, sham::get_queue_details()};
        sycl::buffer<u8> ret = cbuf.copy_back();
        check_buf("copy to host mode", buf_comp, ret);
    }

}


TestStart(Unittest, "shamsys/comm/CommunicationBuffer/send_recv", test_basic_serialized_send_recv, 2){


    u32 nbytes = 1e5;
    sycl::buffer<u8> buf_comp = shamalgs::random::mock_buffer<u8>(0x111, nbytes);


    if(shamcomm::world_rank() == 0){
        shamcomm::CommunicationBuffer cbuf {buf_comp, sham::get_queue_details()};
        MPI_Send(cbuf.get_ptr(), nbytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    }

    if(shamcomm::world_rank() == 1){
        shamcomm::CommunicationBuffer cbuf {nbytes, sham::get_queue_details()};
        MPI_Recv(cbuf.get_ptr(), nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);


        sycl::buffer<u8> ret = cbuf.copy_back();
        check_buf("copy to host mode", buf_comp, ret);
    }


}