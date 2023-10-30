// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeInstance.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "NodeInstance.hpp"

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/EnvVariables.hpp"
#include "shamcomm/CommunicationBuffer.hpp"
#include "shamcomm/details/CommunicationBufferImpl.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamcomm/mpi.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamcomm/mpiInfo.hpp"

#include "MpiDataTypeHandler.hpp"
#include <optional>
#include <stdexcept>
#include <string>



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
        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        for_each_device([&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){

            auto PlatformName = plat.get_info<sycl::info::platform::name>();
            auto DeviceName = dev.get_info<sycl::info::device::name>();


            std::string devname = shambase::trunc_str(DeviceName,29);
            std::string platname = shambase::trunc_str(PlatformName,24);
            std::string devtype = shambase::trunc_str(shambase::getDevice_type(dev),6);

            print_buf += shambase::format("| {:>4} | {:>2} | {:>29.29} | {:>24.24} | {:>6} |",
                rank,key_global,devname,platname,devtype
            ) + "\n";

        });

        std::string recv;
        shamcomm::gather_str(print_buf, recv);
    
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

    u32 compute_queue_id = 0;
    u32 alt_queue_id = 0;

    u32 compute_queue_eu_count = 64;

    u32 get_compute_queue_eu_count(u32 id){
        return compute_queue_eu_count;
    }

    bool is_initialized(){
        int flag = false;
        mpi::initialized(&flag);
        return bool(compute_queue) && bool(alt_queue) && flag;
    };

    void print_queue_map(){
        u32 rank = shamcomm::world_rank();

        std::string print_buf = "";

        std::optional<u32> loc = env::get_local_rank();
        if(loc){
print_buf = shambase::format("| {:>4} | {:>8} | {:>12} | {:>16} |\n", rank,*loc,alt_queue_id,compute_queue_id);
        }else{
print_buf = shambase::format("| {:>4} | {:>8} | {:>12} | {:>16} |\n", rank,"???",alt_queue_id,compute_queue_id);
        }
        

        std::string recv;
        shamcomm::gather_str(print_buf, recv);
    
        if(rank == 0){
            std::string print = "Queue map : \n";
            print+=("----------------------------------------------------\n");
            print+=("| rank | local id | alt queue id | compute queue id |\n");
            print+=("----------------------------------------------------\n");
            print+=(recv);
            print+=("----------------------------------------------------");
            printf("%s\n",print.data());
        }
    }











    












    


    void init(int argc, char *argv[]){

        if(opts::has_option("--sycl-cfg")){

            
            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            //logger::debug_ln("NodeInstance", "chosen sycl config :",sycl_cfg);

            if(shambase::contain_substr(sycl_cfg, "auto:")){

                std::string search = sycl_cfg.substr(5);
                init_auto(search,MPIInitInfo{argc,argv});

            }else {
            
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

            }

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

        shamcomm::fetch_mpi_capabilities();
        
        mpi::init(&mpi_info.argc, &mpi_info.argv);

        shamcomm::fetch_world_info();

        #ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_VALUE:world_size="<<world_size<<"\n";
        std::cout << "%MPI_VALUE:world_rank="<<world_rank<<"\n";
        #endif

        if(shamcomm::world_size() < 1){
            throw ShamsysInstanceException("world size is < 1");
        }

        if(shamcomm::world_rank() < 0){
            throw ShamsysInstanceException("world size is above i32_max");
        }

        int error ;
        //error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
        
        
        if(error != MPI_SUCCESS){
            throw ShamsysInstanceException("failed setting the MPI error mode");
        }

        logger::debug_ln("Sys",
            shambase::format("[{:03}]: \x1B[32mMPI_Init : node n°{:03} | world size : {} | name = {}\033[0m",
            shamcomm::world_rank(),shamcomm::world_rank(),shamcomm::world_size()
            ,get_process_name()));

        mpi::barrier(MPI_COMM_WORLD);
        //if(world_rank == 0){
        if(shamcomm::world_rank() == 0){
            logger::debug_ln("NodeInstance","------------ MPI init ok ------------");
            logger::debug_ln("NodeInstance", "creating MPI type for interop");
        }
        create_sycl_mpi_types();
        if(shamcomm::world_rank() == 0){
            logger::debug_ln("NodeInstance", "MPI type for interop created");
            logger::debug_ln("NodeInstance","------------ MPI / SYCL init ok ------------");
        }
        mpidtypehandler::init_mpidtype();
    }


    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info){

        start_sycl(sycl_info.alt_queue_id, sycl_info.compute_queue_id);

        start_mpi(mpi_info);

    }

    void init_auto(std::string search_key, MPIInitInfo mpi_info){

        start_sycl_auto(search_key);

        start_mpi(mpi_info);

    }

    void close(){

        mpidtypehandler::free_mpidtype();

        if(shamcomm::world_rank() == 0){
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

    void init_queues_auto(std::string search_key){
        std::optional<u32> local_id = env::get_local_rank();

        if(local_id){

            u32 valid_dev_cnt= 0;
            
            details::for_each_device(
            [&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){
                if(shambase::contain_substr(plat.get_info<sycl::info::platform::name>(), search_key)){
                    valid_dev_cnt ++;
                }
            });

            
            u32 valid_dev_id= 0;
            
            details::for_each_device(
            [&](u32 key_global, const sycl::platform & plat, const sycl::device & dev){
                if(shambase::contain_substr(plat.get_info<sycl::info::platform::name>(), search_key)){

                    if((*local_id)%valid_dev_cnt == valid_dev_id){
                        logger::debug_sycl_ln("Sys", "create queue :\n","Local ID :",*local_id,"\n Queue id :",key_global);

                        
                        auto PlatformName = plat.get_info<sycl::info::platform::name>();
                        auto DeviceName = dev.get_info<sycl::info::device::name>();
                        logger::debug_sycl_ln("NodeInstance", "init alt queue  : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                        alt_queue = std::make_unique<sycl::queue>(dev,exception_handler);
                        alt_queue_id = key_global;

                        logger::debug_sycl_ln("NodeInstance", "init comp queue : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                        compute_queue = std::make_unique<sycl::queue>(dev,exception_handler);
                        compute_queue_id = key_global;

                        compute_queue_eu_count = dev.get_info<sycl::info::device::max_compute_units>();
            
                    }

                    valid_dev_id ++;
                }
            });


        }else{
            logger::err_ln("Sys", "cannot query local rank cannot use autodetect");
            throw shambase::throw_with_loc<std::runtime_error>("cannot query local rank cannot use autodetect");
        }
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
                logger::debug_sycl_ln("NodeInstance", "init alt queue  : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                alt_queue = std::make_unique<sycl::queue>(dev,exception_handler);
                alt_queue_id = key_global;
            }

            if(key_global == compute_id){
                logger::debug_sycl_ln("NodeInstance", "init comp queue : ", "|",DeviceName, "|", PlatformName, "|" , shambase::getDevice_type(dev), "|");
                compute_queue = std::make_unique<sycl::queue>(dev,exception_handler);
                compute_queue_id = key_global;

                
                compute_queue_eu_count = dev.get_info<sycl::info::device::max_compute_units>();
                
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

        if(shamcomm::world_rank() == 0){
            logger::debug_ln("Sys", "start sycl queues ...");
        }

        init_queues(alt_id, compute_id);

    }

    void start_sycl_auto(std::string search_key){
        //start sycl

        if(bool(compute_queue) && bool(alt_queue)){
            throw ShamsysInstanceException("Sycl is already initialized");
        }


        init_queues_auto(search_key);

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
        shamcomm::print_mpi_capabilities();
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

        if((shamcomm::mpi_cuda_aware == shamcomm::Yes) && backend == CUDA){
            dgpu_capable = true;
        }

        if((shamcomm::mpi_rocm_aware == shamcomm::Yes) && backend == ROCM){
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
            if(shamcomm::world_rank() == 0){
                logger::warn_ln("Sys", "you are forcing the Direct comm mode to :", force, "it might no work");
            }
            dgpu_mode = dgpu_capable;
        }
    }

    bool is_direct_gpu_selected(){
        return dgpu_mode;
    }



    bool validate_comm(shamcomm::CommunicationProtocol prot){

        u32 nbytes = 1e5;
        sycl::buffer<u8> buf_comp (nbytes);

        {
            sycl::host_accessor acc1 {buf_comp, sycl::write_only, sycl::no_init};
            for(u32 i = 0; i < nbytes; i++){
                acc1[i] = i%100;
            }
        }

        shamcomm::CommunicationBuffer cbuf {buf_comp, prot};
        shamcomm::CommunicationBuffer cbuf_recv {nbytes, prot};

        MPI_Request rq1, rq2;
        if(shamcomm::world_rank() == shamcomm::world_size() -1){
            MPI_Isend(cbuf.get_ptr(), nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &rq1);
        }

        if(shamcomm::world_rank() == 0){
            MPI_Irecv(cbuf_recv.get_ptr(), nbytes, MPI_BYTE, shamcomm::world_size() -1, 0, MPI_COMM_WORLD, &rq2);
        }

        if(shamcomm::world_rank() == shamcomm::world_size() -1){
            MPI_Wait(&rq1, MPI_STATUS_IGNORE);
        }

        if(shamcomm::world_rank() == 0){
            MPI_Wait(&rq2, MPI_STATUS_IGNORE);
        }

        sycl::buffer<u8> recv = shamcomm::CommunicationBuffer::convert(std::move(cbuf_recv));


        bool valid = true;

        if(shamcomm::world_rank() == 0){
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
            if(validate_comm(shamcomm::DirectGPU)){
                if(shamcomm::world_rank() == 0) logger::raw_ln(" - MPI use Direct Comm :",green + "Working"+ terminal_effects::reset);
            }else{
                if(shamcomm::world_rank() == 0)logger::raw_ln(" - MPI use Direct Comm :",red + "Fail"+ terminal_effects::reset);
                if(shamcomm::world_rank() == 0)logger::err_ln("Sys", "the select comm mode failed, try forcing dgpu mode off");
                call_abort = true;
            }
        }else{
            if(validate_comm(shamcomm::CopyToHost)){
                if(shamcomm::world_rank() == 0)logger::raw_ln(" - MPI use Copy to Host :",green + "Working"+ terminal_effects::reset);
            }else{
                if(shamcomm::world_rank() == 0)logger::raw_ln(" - MPI use Copy to Host :",red + "Fail"+ terminal_effects::reset);
                call_abort = true;
            }
        }

        mpi::barrier(MPI_COMM_WORLD);

        if(call_abort){
            MPI_Abort(MPI_COMM_WORLD, 26);
        }
    }

    

}