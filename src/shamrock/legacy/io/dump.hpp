// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file dump.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "aliases.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/patch/utility/merged_patch.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
































inline void file_write_patchdata(MPI_File & mfilepatch , shamrock::patch::PatchData & pdat){

    MPI_Status st;

    std::string head = "##header start##\n";
    head.resize(16);
    mpi::file_write(mfilepatch,head.c_str(), 16, mpi_type_u8, &st);

    head = "#f32";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f32_2";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32_2>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f32_3";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32_3>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f32_4";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32_4>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f32_8";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32_8>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f32_16";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f32_16>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });







    head = "#f64";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f64_2";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64_2>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f64_3";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64_3>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f64_4";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64_4>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f64_8";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64_8>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#f64_16";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<f64_16>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });




    head = "#u32";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<u32>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });

    head = "#u64";
    head.resize(8);
    mpi::file_write(mfilepatch,head.c_str(), 8, mpi_type_u8, &st);

    pdat.for_each_field<u64>([&](auto & a){
        std::string sz = a.get_name();
        u32 obj_cnt = a.get_obj_cnt();
        u32 nvar = a.get_nvar();

        if(sz.size() > 64) throw shambase::throw_with_loc<std::runtime_error>("field name must be shorter than 64 chars");

        sz.resize(64);

        mpi::file_write(mfilepatch,sz.c_str(), 64, mpi_type_u8, &st);
        mpi::file_write(mfilepatch,&nvar, 1, mpi_type_u32, &st);
        mpi::file_write(mfilepatch,&obj_cnt, 1, mpi_type_u32, &st);
    });


    head = "##header end##\n";
    head.resize(16);
    mpi::file_write(mfilepatch,head.c_str(), 16, mpi_type_u8, &st);




    
    #define X(arg) pdat.for_each_field<arg>([&](auto & a){\
        patchdata_field::file_write(mfilepatch, a);\
    });
    XMAC_LIST_ENABLED_FIELD
    #undef X


}

inline void dump_patch_data(std::string prefix, PatchScheduler & sched){

    struct PatchFile{
        MPI_File mfile;
        std::string name;
    };

    std::unordered_map<u64, u64> pfile_map;
    std::vector<PatchFile> patch_files(sched.patch_list.global.size());

    for(u32 i = 0 ; i < sched.patch_list.global.size(); i++){
        pfile_map[sched.patch_list.global[i].id_patch] = i;
        patch_files[i].name = prefix + "patchdata_" + std::to_string(sched.patch_list.global[i].id_patch) + ".bin";
    }
    
    for(PatchFile & pf : patch_files){

        std::cout << "opening : " << pf.name << std::endl;
        int rc = mpi::file_open(MPI_COMM_WORLD, pf.name.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &pf.mfile);

        if (rc) {
            printf( "Unable to open file \"%s\"\n", pf.name.c_str() );fflush(stdout);
        }
    }
    

    {

        sched.patch_data.for_each_patchdata([&](u64 pid, shamrock::patch::PatchData & pdat){

            std::cout << "[" << shamsys::instance::world_rank << "] writing pdat : " << pid << std::endl;
            
            MPI_Status st;
            MPI_File & mfilepatch = patch_files[pfile_map[pid]].mfile;


            file_write_patchdata(mfilepatch, pdat);

        });


        
    }

    for(PatchFile & pf : patch_files){
        mpi::file_close(&pf.mfile);
    }
}

inline void dump_patch_list(std::string prefix, PatchScheduler & sched){
    

    MPI_File patch_list_file;
    std::string fname = prefix + "patch_list.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &patch_list_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(shamsys::instance::world_rank == 0){

        if(sched.patch_list.global.size() > u64(u32_max)){
            throw shambase::throw_with_loc<std::runtime_error>("patch list size > u32_max not handled by dump");
        }

        MPI_Status st;
        mpi::file_write(patch_list_file, sched.patch_list.global.data(), sched.patch_list.global.size(), shamrock::patch::get_patch_mpi_type<3>(),&st);
    }



    mpi::file_close(&patch_list_file);
}

inline void dump_simbox(std::string prefix, PatchScheduler & sched){
    MPI_File simbox_file;
    std::string fname = prefix + "simbox.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &simbox_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(shamsys::instance::world_rank == 0){


        MPI_Status st;

        if(sched.pdl.check_main_field_type<f32_3>()){
            u8 f = 0;
            auto [bmin, bmax] = sched.patch_data.sim_box.get_bounding_box<f32_3>();
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &bmin, 1, mpi_type_f32_3,&st);
            mpi::file_write(simbox_file, &bmax, 1, mpi_type_f32_3,&st);
        }else if(sched.pdl.check_main_field_type<f64_3>()){
            u8 f = 1;
            auto [bmin, bmax] = sched.patch_data.sim_box.get_bounding_box<f64_3>();
            mpi::file_write(simbox_file, &f, 1,mpi_type_u8,&st);
            mpi::file_write(simbox_file, &bmin, 1, mpi_type_f64_3,&st);    
            mpi::file_write(simbox_file, &bmax, 1, mpi_type_f64_3,&st);       
        }
        
    }



    mpi::file_close(&simbox_file);
}

inline void dump_siminfo(std::string prefix, f64 time){
    MPI_File timeval_file;
    std::string fname = prefix + "timeval.bin";

    int rc = mpi::file_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &timeval_file);

    if (rc) {
        printf( "Unable to open file \"%s\"\n", fname.c_str() );fflush(stdout);
    }



    if(shamsys::instance::world_rank == 0){


        MPI_Status st;
        mpi::file_write(timeval_file, &time, 1,mpi_type_f64,&st);

        
    }



    mpi::file_close(&timeval_file);
}

inline void dump_state(std::string prefix, PatchScheduler & sched, f64 time){StackEntry stack_loc{};

    dump_patch_data(prefix, sched);

    dump_patch_list(prefix, sched);
    dump_simbox(prefix, sched);
    dump_siminfo(prefix,time);


}



