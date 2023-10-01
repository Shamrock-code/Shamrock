// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good



#include "shamtest.hpp"
#include <cstdlib>
#include <filesystem>
#include <pybind11/embed.h>
#include <sstream>

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"

#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/MpiWrapper.hpp"

#include "shambase/exception.hpp"

#include "shambindings/pybindaliases.hpp"


bool has_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }
    
    return false;
}

std::string_view get_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }
    
    return "";
}



namespace shamtest {
    int run_all_tests(int argc, char *argv[], bool run_bench,bool run_analysis, bool run_unittest){
        StackEntry stack {};
        
    
        using namespace shamtest::details;



        if(opts::has_option("--test-list")){


            auto print_list = [&](TestType t){
                for (auto test : static_init_vec_tests) {
                    if (test.type == t) {
                        if(test.node_count == -1){
                            printf("- [any] %-15s\n",test.name.c_str());
                        }else{
                            printf("- [%03d] %-15s\n",test.node_count,test.name.c_str());
                        }
                    }
                }
            };

            printf("--- Benchmark ---\n");

            print_list(Benchmark);

            printf("--- Analysis  ---\n");

            print_list(Analysis);

            printf("--- Unittest  ---\n");

            print_list(Unittest);

            
            return 0;
        }

        bool run_only = false;
        std::string run_only_name = "";
        if(opts::has_option("--run-only")){
            run_only_name = opts::get_option("--run-only");
            run_only = true;
        }


        bool full_output = opts::has_option("--full-output");

        bool out_to_file = opts::has_option("-o");

        if(out_to_file){
            if(opts::get_option("-o").size() == 0){
                opts::print_help();
            }
        }


        using namespace shamsys;


        if(!run_only){
            printf("\n------------ Tests list --------------\n");
        }

        std::vector<u32> selected_tests = {};


        auto can_run = [&](shamtest::details::Test & t) -> bool {

            bool any_node_cnt = (t.node_count == -1);
            bool world_size_ok = t.node_count == instance::world_size;

            bool can_run_type = false;

            auto test_type = t.type;
            can_run_type = can_run_type || (run_unittest && (Unittest == test_type)) ;
            can_run_type = can_run_type || (run_analysis && (Analysis == test_type)) ;
            can_run_type = can_run_type || (run_bench && (Benchmark == test_type)) ;

            return can_run_type && (any_node_cnt || world_size_ok );
        };

        auto print_test = [&](shamtest::details::Test & t, bool enabled){
            bool any_node_cnt = (t.node_count == -1);

            if (enabled) {
            
                if(any_node_cnt){
                    printf(" - [\033[;32many\033[0m] ");
                }else{
                    printf(" - [\033[;32m%03d\033[0m] ",t.node_count);
                }
                std::cout << "\033[;32m"<<t.name <<  "\033[0m " <<std::endl;

            }else{
                if(any_node_cnt){
                    printf(" - [\033[;31many\033[0m] ");
                }else{
                    printf(" - [\033[;31m%03d\033[0m] ",t.node_count);
                }
                std::cout << "\033[;31m"<<t.name <<  "\033[0m " <<std::endl;
            }


        };

        

        

        if(run_only){
            
            for (u32 i = 0; i < static_init_vec_tests.size(); i++) {

                bool run_test = can_run(static_init_vec_tests[i]);
                if(run_only_name.compare(static_init_vec_tests[i].name) ==0){
                    if(run_test) {
                        selected_tests.push_back(i);
                    }else{
                        logger::err_ln("TEST", "test can not run under the following configuration");
                    }
                }
            }

        }else{

            auto test_loop = [&](TestType t){
                for (u32 i = 0; i < static_init_vec_tests.size(); i++) {

                    if(static_init_vec_tests[i].type == t){
                        bool run_test = can_run(static_init_vec_tests[i]);
                        print_test(static_init_vec_tests[i], run_test);
                        if(run_test) {selected_tests.push_back(i);}
                    }
                    
                }
            };

            if(run_bench){
                printf("--- Benchmark ---\n");
                test_loop(Benchmark);
            }

            if(run_analysis){
                printf("--- Analysis  ---\n");
                test_loop(Analysis);
            }

            if(run_unittest){
                printf("--- Unittest  ---\n");
                test_loop(Unittest);
            }
            
        }

        if(!run_only){
            printf("--------------------------------------\n\n");
        }



        std::stringstream rank_test_res_out;




        u32 test_loc_cnt = 0;

        bool has_error = false;

        logger::info_ln("Test", "start python interpreter");
        py::initialize_interpreter();

        std::filesystem::create_directories("tests/figures");

        for (u32 i : selected_tests) {

            shamtest::details::Test & test = static_init_vec_tests[i];

            if(run_only){
                printf("running test : ");
            }else{
                printf("running test [%d/%zu] : ",test_loc_cnt+1,selected_tests.size());
            }

            bool any_node_cnt = test.node_count == -1;
            if(any_node_cnt){
                printf(" [any] ");
            }else{
                printf(" [%03d] ",test.node_count);
            }

            std::cout << "\033[;34m"<<test.name <<  "\033[0m " <<std::endl;
            




            mpi::barrier(MPI_COMM_WORLD);
            shambase::Timer timer;
            timer.start();
            auto res = test.run();
            timer.end();
            mpi::barrier(MPI_COMM_WORLD);

            
            
            for(unsigned int j = 0; j < res.asserts.asserts.size(); j++){


                if(full_output || (!res.asserts.asserts[j].value)){
                    printf("        [%d/%zu] : ",j+1,res.asserts.asserts.size());
                    printf("%-20s",res.asserts.asserts[j].name.c_str());
                    
                    if(res.asserts.asserts[j].value){
                        std::cout << "  (\033[;32mSucces\033[0m)\n";
                    }else{
                        has_error = true;
                        std::cout << "  (\033[1;31m Fail \033[0m)\n";
                        if(! res.asserts.asserts[j].comment.empty()){
                            std::cout << "----- logs : \n" << res.asserts.asserts[j].comment << "\n-----" << std::endl;
                        }
                    }
                }
            }

                
            

            u32 succes_cnt = 0;
            for(unsigned int j = 0; j < res.asserts.asserts.size(); j++){
                if(res.asserts.asserts[j].value){
                    succes_cnt++;
                }
            }
            
            if(succes_cnt == res.asserts.asserts.size()){
                std::cout << "       Result : \033[;32mSucces\033[0m";
            }else{
                std::cout << "       Result : \033[1;31m Fail \033[0m";
            }



            std::string s_assert = shambase::format(" [{}/{}] ",succes_cnt,res.asserts.asserts.size());
            printf("%-15s",s_assert.c_str());
            std::cout << " (" << timer.get_time_str() << ")" <<std::endl;

            
            std::cout << std::endl;

            rank_test_res_out << res.serialize() << ",";


            test_loc_cnt++;

        }


        logger::info_ln("Test", "close python interpreter");
        py::finalize_interpreter();



        //recover result on node 0 and write to file if -o <outfile> specified
        std::string out_res_string;

        if(instance::world_size == 1){
            out_res_string = rank_test_res_out.str();
        }else{
            std::string loc_string = rank_test_res_out.str();


            //printf("sending : \n%s\n",loc_string.c_str());

            int *counts = new int[instance::world_size];
            int nelements = (int) loc_string.size();
            // Each process tells the root how many elements it holds
            mpi::gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);


            // Displacements in the receive buffer for MPI_GATHERV
            int *disps = new int[instance::world_size];
            // Displacement for the first chunk of data - 0
            for (int i = 0; i < instance::world_size; i++)
                disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

            // Place to hold the gathered data
            // Allocate at root only
            char *gather_data = NULL;
            if (instance::world_rank == 0)
                // disps[size-1]+counts[size-1] == total number of elements
                gather_data = new char[disps[instance::world_size-1]+counts[instance::world_size-1]];


            // Collect everything into the root
            mpi::gatherv(loc_string.c_str(), nelements, MPI_CHAR,
                        gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

            
            if(instance::world_rank == 0){
                out_res_string = std::string(gather_data,disps[instance::world_size-1]+counts[instance::world_size-1]);
            }

        }
        

        //generate json output and write it into the specified file
        if(instance::world_rank == 0){

            if(out_res_string.back() == ','){
                out_res_string = out_res_string.substr(0,out_res_string.size()-1);
            }

            std::string s_out;

            s_out = "{\n";

            s_out += R"(    "commit_hash" : ")" + git_commit_hash + "\",\n" ;
            s_out += R"(    "world_size" : ")" + std::to_string(instance::world_size) + "\",\n" ;

            #if defined (SYCL_COMP_INTEL_LLVM)
            s_out += R"(    "compiler" : "DPCPP",)" "\n" ;
            #elif defined (SYCL_COMP_HIPSYCL)
            s_out += R"(    "compiler" : "HipSYCL",)" "\n" ;
            #else
            s_out += R"(    "compiler" : "Unknown",)" "\n" ;
            #endif  

            s_out += R"(    "comp_args" : ")" + compile_arg + "\",\n" ;


            s_out += R"(    "results" : )"   "[\n\n" ;
            s_out += shambase::increase_indent(out_res_string);
            s_out += "\n    ]\n}";

            //printf("%s\n",s_out.c_str());

            if(out_to_file){
                shambase::write_string_to_file(std::string(opts::get_option("-o")), s_out);
            }
            
        }


        instance::close();


        if(has_error){
            return -1;
        }else{
            return 0;
        }
    }
}