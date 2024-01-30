// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/memory.hpp"
#include "shamalgs/reduction.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamalgs/collective/sparseXchg.hpp"
#include "shamalgs/random.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

void sparse_comm_test(std::string prefix, sham::queues::QueueDetails & qdet){
    
    using namespace shamalgs::collective;
    using namespace shamsys::instance;
    using namespace shamsys;
    using namespace shamcomm;

    const i32 wsize = world_size();
    const i32 wrank = world_rank();
    
    u32 num_buf = wsize*5;
    u32 nbytes_per_buf = 1e4;

    u64 seed = 0x123;
    std::mt19937 eng(seed);

    struct RefBuff{
        i32 sender_rank;
        i32 receiver_rank;
        std::unique_ptr<sycl::buffer<u8>> payload;
    };

    struct TestElements{
        std::vector<RefBuff> elements;

        void add_element(std::mt19937 & eng, u32 wsize, u64 bytes){
            u64 rnd = eng();
            elements.push_back(RefBuff{
                shamalgs::random::mock_value<i32>(eng, 0,wsize-1),
                shamalgs::random::mock_value<i32>(eng, 0,wsize-1),
                std::make_unique<sycl::buffer<u8>>(
                    shamalgs::random::mock_buffer<u8>(rnd, 
                    shamalgs::random::mock_value<i32>(eng, 1,bytes)))
            });
        }

        void sort_input(){
            std::sort(elements.begin(),elements.end(),[] (const auto& lhs, const auto& rhs) {
                return lhs.sender_rank < rhs.sender_rank;
            });
        }
    };

    TestElements tests;
    for(u32 i = 0; i < num_buf; i++){
        tests.add_element(eng, wsize, nbytes_per_buf);
    }
    tests.sort_input();


    //make comm bufs

    std::vector<SendPayload> sendop;

    for(RefBuff & bufinfo : tests.elements){
        if(bufinfo.sender_rank == world_rank()){
            sendop.push_back(SendPayload{
                bufinfo.receiver_rank,
                std::make_unique<CommunicationBuffer>(*bufinfo.payload, qdet)
            });
        }
    }


    std::vector<RecvPayload> recvop;
    base_sparse_comm(sendop, recvop);

    std::vector<RefBuff> recv_data;
    for(RecvPayload & load : recvop){
        recv_data.push_back(RefBuff{
            load.sender_ranks,
            wrank,
            std::make_unique<sycl::buffer<u8>>(load.payload->copy_back())
        });
    }

    logger::raw_ln("ref data : ");
    for (RefBuff & ref : tests.elements) {
        logger::raw_ln(
            shambase::format("[{:2}] {} -> {} ({})", 
            wrank,
            ref.sender_rank,
            ref.receiver_rank, 
            ref.payload->size()
            ));
    }

    logger::raw_ln("recv data : ");
    for (RefBuff & ref : recv_data) {
        logger::raw_ln(
            shambase::format("[{:2}] {} -> {} ({})", 
            wrank,
            ref.sender_rank,
            ref.receiver_rank, 
            ref.payload->size()
            ));
    }

    u32 ref_idx = 0;
    for (RefBuff & ref : tests.elements) {
        if(ref.receiver_rank == wrank){

            if(ref_idx < recv_data.size()){

                RefBuff & recv_buf = recv_data[ref_idx];

                shamtest::asserts().assert_equal(prefix+"same sender", recv_buf.sender_rank , ref.sender_rank);
                shamtest::asserts().assert_equal(prefix+"same receiver", recv_buf.receiver_rank , ref.receiver_rank);
                shamtest::asserts().assert_bool(prefix+"same buffer", shamalgs::reduction::equals_ptr(ref.payload, recv_buf.payload));

            }else{
                throw shambase::make_except_with_loc<std::runtime_error>(prefix+"missing recv mesages");
            }

            ref_idx ++;
        }
    }
}


TestStart(Unittest, "shamalgs/collective/sparseXchg", testsparsexchg, -1){

    sparse_comm_test("",sham::get_queue_details());

}