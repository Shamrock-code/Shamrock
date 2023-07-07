// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "NodeInstance.hpp"

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/EnvVariables.hpp"
#include "shamsys/comm/CommunicationBuffer.hpp"
#include "shamsys/comm/details/CommunicationBufferImpl.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/MpiWrapper.hpp"
#include <hipSYCL/sycl/info/platform.hpp>
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <mpi-ext.h>
#include "shamsys/legacy/sycl_mpi_interop.hpp"

#include "MpiDataTypeHandler.hpp"
#include <mpi.h>
#include <stdexcept>
#include <string>


namespace shamsys::instance::tmpmpi{
    /**
     * @brief allgatherv on vector with size query (size querrying variant of vector_allgatherv_ks)
     * //TODO add fault tolerance
     * @tparam T 
     * @param send_vec 
     * @param send_type 
     * @param recv_vec 
     * @param recv_type 
     */
    inline void gather_str(const std::string & send_vec ,std::string & recv_vec){
        StackEntry stack_loc{};

        u32 local_count = send_vec.size();

        //querry global size and resize the receiving vector
        u32 global_len;
        mpi::allreduce(&local_count, &global_len, 1, MPI_INT , MPI_SUM, MPI_COMM_WORLD);
        recv_vec.resize(global_len);

        int* table_data_count = new int[shamsys::instance::world_size];

        mpi::allgather(
            &local_count, 
            1, 
            MPI_INT, 
            &table_data_count[0], 
            1, 
            MPI_INT, 
            MPI_COMM_WORLD);

        //printf("table_data_count = [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);



        int* node_displacments_data_table = new int[shamsys::instance::world_size];

        node_displacments_data_table[0] = 0;

        for(u32 i = 1 ; i < shamsys::instance::world_size; i++){
            node_displacments_data_table[i] = node_displacments_data_table[i-1] + table_data_count[i-1];
        }
        
        //printf("node_displacments_data_table = [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);
        
        mpi::allgatherv(
            send_vec.data(), 
            send_vec.size(),
            MPI_CHAR, 
            recv_vec.data(), 
            table_data_count, 
            node_displacments_data_table, 
            MPI_CHAR, 
            MPI_COMM_WORLD);


        delete [] table_data_count;
        delete [] node_displacments_data_table;
    } 
}

namespace shamsys::mpi_features{

    enum StateMPI_Aware{
        Unknown, Yes, No
    };

    StateMPI_Aware mpi_cuda_aware;
    StateMPI_Aware mpi_rocm_aware;

    void fetch_mpi_capabilities(){
        //detect MPI cuda aware
        #if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
            mpi_cuda_aware = Yes;
        } else{
            mpi_cuda_aware = No;
        }
        #else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        mpi_cuda_aware = Unknown;
        #endif /* MPIX_CUDA_AWARE_SUPPORT */ 

        //detect MPI rocm aware
        #if defined(MPIX_ROCM_AWARE_SUPPORT)
        if (1 == MPIX_Query_rocm_support()) {
            mpi_rocm_aware = Yes;
        }else{
            mpi_rocm_aware = No;
        }
        #else  /* !defined(MPIX_ROCM_AWARE_SUPPORT) */
        mpi_rocm_aware = Unknown;
        #endif /* MPIX_ROCM_AWARE_SUPPORT */ 
    }
    
    void print_mpi_capabilities(){
        using namespace terminal_effects::colors_foreground_8b;
        if (mpi_cuda_aware == Yes) {
            logger::raw_ln(" - MPI CUDA-AWARE :",green + "Yes"+ terminal_effects::reset);
        }else if (mpi_cuda_aware == No) {
            logger::raw_ln(" - MPI CUDA-AWARE :",red + "No"+ terminal_effects::reset);
        }else if (mpi_cuda_aware == Unknown) {
            logger::raw_ln(" - MPI CUDA-AWARE :",yellow + "Unknown"+ terminal_effects::reset);
        }

        if (mpi_rocm_aware == Yes) {
            logger::raw_ln(" - MPI ROCM-AWARE :",green + "Yes"+ terminal_effects::reset);
        }else if (mpi_rocm_aware == No) {
            logger::raw_ln(" - MPI ROCM-AWARE :",red + "No"+ terminal_effects::reset);
        }else if (mpi_rocm_aware == Unknown) {
            logger::raw_ln(" - MPI ROCM-AWARE :",yellow + "Unknown"+ terminal_effects::reset);
        }
    }

}

namespace shamsys::instance::details {

    

    /**
     * @brief validate a sycl queue
     * 
     * @param q 
     */
    void check_queue_is_valid(sycl::queue & q){

        auto test_kernel = [](sycl::queue & q){
            sycl::buffer<u32> b(10);

            q.submit([&](sycl::handler & cgh){
                sycl::accessor acc {b, cgh,sycl::write_only,sycl::no_init};

                cgh.parallel_for(sycl::range<1>{10},[=](sycl::item<1> i){
                    acc[i] = i.get_linear_id();
                });
            });

            q.wait();

            {
                sycl::host_accessor acc {b, sycl::read_only};
                if(acc[9] != 9){
                    throw shambase::throw_with_loc<std::runtime_error>("The chosen SYCL queue cannot execute a basic kernel");
                }
            }
        };


        std::exception_ptr eptr;
        try {
            test_kernel(q);
            //logger::info_ln("NodeInstance", "selected queue :",q.get_device().get_info<sycl::info::device::name>()," working !");
        } catch(...) {
            eptr = std::current_exception(); // capture
        }

        if (eptr) {
            //logger::err_ln("NodeInstance", "selected queue :",q.get_device().get_info<sycl::info::device::name>(),"does not function properly");
            std::rethrow_exception(eptr);
        }


    }

    /**
     * @brief for each SYCL device
     * 
     * @param fct 
     * @return u32 the number of devices
     */
    u32 for_each_device(std::function<void(u32, const sycl::platform &, const sycl::device &)> fct){

        u32 key_global = 0;
        const auto &Platforms = sycl::platform::get_platforms();    
        for (const auto &Platform : Platforms) {
            const auto &Devices = Platform.get_devices();
            for (const auto &Device : Devices) {
                fct(key_global, Platform, Device);   
                key_global ++;
            }
        }
        return key_global;
    }

    void print_device_list(){
        u32 rank = world_rank;

        std::string print_buf = "";

        for_each_device([&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){

            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName = dev.get_info<sycl::info::device::name>();

            std::string devname = shambase::trunc_str(DeviceName,29);
            std::string platname = shambase::trunc_str(PlatformName,24);
            std::string devtype = shambase::trunc_str(shambase::getDevice_type(dev),6);

            print_buf += shambase::format("| {:>4} | {:>2} | {:>29.29} | {:>24.24} | {:>6} |",
                rank,key_global,devname.c_str(),platname.c_str(),devtype.c_str()
            ) + "\n";

        });

        std::string recv;
        tmpmpi::gather_str(print_buf, recv);
    
        if(rank == 0){
            std::string print = "Cluster SYCL Info : \n";
            print+=("--------------------------------------------------------------------------------\n");
            print+=("| rank | id |        Device name            |       Platform name      |  Type  |\n");
            print+=("--------------------------------------------------------------------------------\n");
            print+=(recv);
            print+=("--------------------------------------------------------------------------------");
            printf("%s\n",print.data());
        }
    }



} // namespace shamsys::instance::details


namespace shamsys::instance {

    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                printf("Caught synchronous SYCL exception: %s\n",e.what());
            }
        }
    };


    std::unique_ptr<sycl::queue> compute_queue;
    std::unique_ptr<sycl::queue> alt_queue;

    












    












    bool is_initialized(){
        int flag = false;
        mpi::initialized(&flag);
        return bool(compute_queue) && bool(alt_queue) && flag;
    };


    void init(int argc, char *argv[]){


        if(opts::has_option("--sycl-cfg")){
            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            //logger::debug_ln("NodeInstance", "chosen sycl config :",sycl_cfg);

            size_t split_alt_comp = 0;
            split_alt_comp = sycl_cfg.find(":");

            if(split_alt_comp == std::string::npos){
                logger::err_ln("NodeInstance", "sycl-cfg layout should be x:x");
                throw ShamsysInstanceException("sycl-cfg layout should be x:x");
            }

            std::string alt_cfg = sycl_cfg.substr(0, split_alt_comp);
            std::string comp_cfg = sycl_cfg.substr(split_alt_comp+1, sycl_cfg.length());


            u32 ialt, icomp;
            try {
                try {
                    ialt = std::stoi(alt_cfg);
                } catch (const std::invalid_argument & a) {
                    logger::err_ln("NodeInstance", "alt config is not an int");
                    throw ShamsysInstanceException("alt config is not an int");
                }
            } catch (const std::out_of_range & a) {
                logger::err_ln("NodeInstance", "alt config is to big for an integer");
                throw ShamsysInstanceException("alt config is to big for an integer");
            }

            try {
                try {
                    icomp = std::stoi(comp_cfg);
                } catch (const std::invalid_argument & a) {
                    logger::err_ln("NodeInstance", "compute config is not an int");
                    throw ShamsysInstanceException("compute config is not an int");
                }
            } catch (const std::out_of_range & a) {
                logger::err_ln("NodeInstance", "compute config is to big for an integer");
                throw ShamsysInstanceException("compute config is to big for an integer");
            }

            init(SyclInitInfo{ialt,icomp}, MPIInitInfo{argc,argv});

        }else {

            logger::err_ln("NodeInstance", "Please specify a sycl configuration (--sycl-cfg x:x)");
            //std::cout << "[NodeInstance] Please specify a sycl configuration (--sycl-cfg x:x)" << std::endl;
            throw ShamsysInstanceException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }

    }

    





    

    void start_mpi( MPIInitInfo mpi_info){

        #ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_DEFINE:MPI_COMM_WORLD="<<MPI_COMM_WORLD<<"\n";
        #endif

        shamsys::mpi_features::fetch_mpi_capabilities();
        
        mpi::init(&mpi_info.argc, &mpi_info.argv);

        i32 iworld_size, iworld_rank;

        mpi::comm_size(MPI_COMM_WORLD, &iworld_size);
        mpi::comm_rank(MPI_COMM_WORLD, &iworld_rank);

        world_rank = iworld_rank;
        world_size = iworld_size;

        #ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_VALUE:world_size="<<world_size<<"\n";
        std::cout << "%MPI_VALUE:world_rank="<<world_rank<<"\n";
        #endif

        if(world_size < 1){
            throw ShamsysInstanceException("world size is < 1");
        }

        if(world_rank < 0){
            throw ShamsysInstanceException("world size is above i32_max");
        }

        int error ;
        //error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
        
        
        if(error != MPI_SUCCESS){
            throw ShamsysInstanceException("failed setting the MPI error mode");
        }

        logger::debug_ln("Sys",shambase::format("[{:03}]: \x1B[32mMPI_Init : node n°{:03} | world size : {} | name = {}\033[0m",world_rank,world_rank,world_size,get_process_name().c_str()));

        mpi::barrier(MPI_COMM_WORLD);
        //if(world_rank == 0){
        if(world_rank == 0){
            logger::debug_ln("------------ MPI init ok ------------ \n");
            logger::debug_ln("NodeInstance", "creating MPI type for interop");
        }
        create_sycl_mpi_types();
        if(world_rank == 0){
            logger::debug_ln("NodeInstance", "MPI type for interop created");
            logger::debug_ln("------------ MPI / SYCL init ok ------------ \n");
        }
        mpidtypehandler::init_mpidtype();
    }


    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info){

        start_sycl(sycl_info.alt_queue_id, sycl_info.compute_queue_id);

        start_mpi(mpi_info);

    }

    void close(){

        mpidtypehandler::free_mpidtype();

        if(world_rank == 0){
            logger::print_faint_row();
            logger::raw_ln(" - MPI finalize \nExiting ...\n");
            logger::raw_ln(" Hopefully it was quick :')\n");
        }
        mpi::finalize(); 

        alt_queue.reset();
        compute_queue.reset();
    }


















    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    sycl::queue & get_compute_queue(u32  /*id*/){
        if(!compute_queue){ throw ShamsysInstanceException("sycl handler is not initialized");}
        return * compute_queue;
    }

    sycl::queue & get_alt_queue(u32  /*id*/){
        if(!alt_queue){ throw ShamsysInstanceException("sycl handler is not initialized");}
        return * alt_queue;
    }

    void print_device_info(const sycl::device &Device){
        std::cout 
            << "   - " 
            << Device.get_info<sycl::info::device::name>()
            << " " 
            << shambase::readable_sizeof(Device.get_info<sycl::info::device::global_mem_size>()) << "\n";
    }

    void print_device_list(){
        details::print_device_list();
    }

    void init_queues(u32 alt_id, u32 compute_id){


        u32 cnt_dev = details::for_each_device(
            [&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){});

        if(alt_id >= cnt_dev){
            throw shambase::throw_with_loc<std::invalid_argument>("the alt queue id is larger than the number of queue");
        }

        if(compute_id >= cnt_dev){
            throw shambase::throw_with_loc<std::invalid_argument>("the compute queue id is larger than the number of queue");
        }

        details::for_each_device([&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){

            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName = dev.get_info<sycl::info::device::name>();

            if(key_global == alt_id){
                logger::debug_ln("NodeInstance", "init alt queue  : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                alt_queue = std::make_unique<sycl::queue>(dev,exception_handler);
            }

            if(key_global == compute_id){
                logger::debug_ln("NodeInstance", "init comp queue : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                compute_queue = std::make_unique<sycl::queue>(dev,exception_handler);
            }

        });

        details::check_queue_is_valid(*compute_queue);
        details::check_queue_is_valid(*alt_queue);
    }


    void start_sycl(u32 alt_id, u32 compute_id){
        //start sycl

        if(bool(compute_queue) && bool(alt_queue)){
            throw ShamsysInstanceException("Sycl is already initialized");
        }

        if(world_rank == 0){
            logger::debug_ln("Sys", "start sycl queues ...");
        }

        init_queues(alt_id, compute_id);

    }


    ////////////////////////////
    //MPI related routines
    ////////////////////////////


    std::string get_process_name(){

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;

        int err_code = mpi::get_processor_name(processor_name, &name_len);

        if(err_code != MPI_SUCCESS){
            throw ShamsysInstanceException("failed getting the process name");
        }
        
        return std::string(processor_name);
    }


    void print_mpi_capabilities(){
        shamsys::mpi_features::print_mpi_capabilities();
    }


    bool dgpu_mode = false;
    bool dgpu_capable = false;
    void check_dgpu_available(){

        dgpu_capable = false;

        enum backenddevice{
            CUDA, ROCM, Unknown, OpenMP
        } backend = Unknown;

        std::string pname = get_compute_queue()
            .get_device()
            .get_platform()
            .get_info<sycl::info::platform::name>();

        if(shambase::contain_substr(pname, "CUDA")){backend = CUDA;}
        if(shambase::contain_substr(pname, "NVIDIA")){backend = CUDA;}
        if(shambase::contain_substr(pname, "ROCM")){backend = ROCM;}
        if(shambase::contain_substr(pname, "AMD")){backend = ROCM;}
        if(shambase::contain_substr(pname, "OpenMP")){backend = OpenMP;}

        if((shamsys::mpi_features::mpi_cuda_aware == mpi_features::Yes) && backend == CUDA){
            dgpu_capable = true;
        }

        if((shamsys::mpi_features::mpi_rocm_aware == mpi_features::Yes) && backend == ROCM){
            dgpu_capable = true;
        }

        if(backend == OpenMP){
            dgpu_capable = true;
        }

        using namespace terminal_effects::colors_foreground_8b;
        if(dgpu_capable){
            logger::raw_ln(" - MPI use Direct Comm :",green + "Yes"+ terminal_effects::reset);
        }else{
            logger::raw_ln(" - MPI use Direct Comm :",red + "No"+ terminal_effects::reset);
        }
        dgpu_mode = dgpu_capable;
    }

    void force_direct_gpu_mode(bool force){
        if(force != dgpu_capable){
            if(world_rank == 0){
                logger::warn_ln("Sys", "you are forcing the Direct comm mode to :", force, "it might no work");
            }
            dgpu_mode = dgpu_capable;
        }
    }

    bool is_direct_gpu_selected(){
        return dgpu_mode;
    }



    bool validate_comm(shamsys::CommunicationProtocol prot){

        u32 nbytes = 1e5;
        sycl::buffer<u8> buf_comp (nbytes);

        {
            sycl::host_accessor acc1 {buf_comp, sycl::write_only, sycl::no_init};
            for(u32 i = 0; i < nbytes; i++){
                acc1[i] = i%100;
            }
        }

        shamsys::CommunicationBuffer cbuf {buf_comp, prot};
        shamsys::CommunicationBuffer cbuf_recv {nbytes, prot};

        MPI_Request rq1, rq2;
        if(world_rank == world_size -1){
            MPI_Isend(cbuf.get_ptr(), nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &rq1);
        }

        if(world_rank == 0){
            MPI_Irecv(cbuf_recv.get_ptr(), nbytes, MPI_BYTE, world_size -1, 0, MPI_COMM_WORLD, &rq2);
        }

        if(world_rank == world_size -1){
            MPI_Wait(&rq1, MPI_STATUS_IGNORE);
        }

        if(world_rank == 0){
            MPI_Wait(&rq2, MPI_STATUS_IGNORE);
        }

        sycl::buffer<u8> recv = shamsys::CommunicationBuffer::convert(std::move(cbuf_recv));


        bool valid = true;

        if(world_rank == 0){
            sycl::host_accessor acc1 {buf_comp};
            sycl::host_accessor acc2 {recv};

            std::string id_err_list = "errors in id : ";

            bool eq = true;
            for(u32 i = 0; i < recv.size(); i++){
                if(!shambase::vec_equals(acc1[i] , acc2[i])){
                    eq = false;
                    //id_err_list += std::to_string(i) + " ";
                }
            }

            valid = eq;
        }


        return valid;
    }

    void validate_comm(){
        u32 nbytes = 1e5;
        sycl::buffer<u8> buf_comp (nbytes);

        bool call_abort = false;

        using namespace terminal_effects::colors_foreground_8b;
        if(dgpu_mode){
            if(validate_comm(shamsys::DirectGPU)){
                if(world_rank == 0) logger::raw_ln(" - MPI ues Direct Comm :",green + "Working"+ terminal_effects::reset);
            }else{
                if(world_rank == 0)logger::raw_ln(" - MPI use Direct Comm :",red + "Fail"+ terminal_effects::reset);
                if(world_rank == 0)logger::err_ln("Sys", "the select comm mode failed, try forcing dgpu mode off");
                call_abort = true;
            }
        }else{
            if(validate_comm(shamsys::CopyToHost)){
                if(world_rank == 0)logger::raw_ln(" - MPI ues Copy to Host :",green + "Working"+ terminal_effects::reset);
            }else{
                if(world_rank == 0)logger::raw_ln(" - MPI use Copy to Host :",red + "Fail"+ terminal_effects::reset);
                call_abort = true;
            }
        }

        mpi::barrier(MPI_COMM_WORLD);

        if(call_abort){
            MPI_Abort(MPI_COMM_WORLD, 26);
        }
    }

    

}