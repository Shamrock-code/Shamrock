// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"

#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shammodels/sph/models/basic_sph_gas.hpp"
#include "shammodels/sph/setup/sph_setup.hpp"
#include "shamtest/shamtest.hpp"







template<class flt> 
std::tuple<f64,f64,f64> benchmark_periodic_box(f32 dr, u32 npatch){

    using vec = sycl::vec<flt,3>;

    u64 Nesti = (2.F/dr)*(2.F/dr)*(2.F/dr);

    PatchDataLayout pdl;

    pdl.xyz_mode = xyz32;
    pdl.add_field<f32_3>("xyz", 1);
    pdl.add_field<f32>("hpart", 1);
    pdl.add_field<f32_3>("vxyz",1);
    pdl.add_field<f32_3>("axyz",1);
    pdl.add_field<f32_3>("axyz_old",1);

    auto id_v = pdl.get_field_idx<f32_3>("vxyz");
    auto id_a = pdl.get_field_idx<f32_3>("axyz");

    PatchScheduler sched = PatchScheduler(pdl,Nesti/npatch, 1);
    sched.init_mpi_required_types();

    auto setup = [&]() -> std::tuple<flt,f64>{
        using Setup = models::sph::SetupSPH<f32, models::sph::kernels::M4<f32>>;

        Setup setup;
        setup.init(sched);

        auto box = setup.get_ideal_box(dr, {vec{-1,-1,-1},vec{1,1,1}});

        //auto ebox = box;
        //std::get<0>(ebox).x() -= 1e-5;
        //std::get<0>(ebox).y() -= 1e-5;
        //std::get<0>(ebox).z() -= 1e-5;
        //std::get<1>(ebox).x() += 1e-5;
        //std::get<1>(ebox).y() += 1e-5;
        //std::get<1>(ebox).z() += 1e-5;
        sched.set_box_volume<f32_3>(box);


        setup.set_boundaries(true);
        setup.add_particules_fcc(sched, dr, box);
        setup.set_total_mass(8.);
        
        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            pdat.get_field<f32_3>(id_v).override(f32_3{0,0,0});
            pdat.get_field<f32_3>(id_a).override(f32_3{0,0,0});
        });



        sched.scheduler_step(true, true);

        sched.scheduler_step(true, true);

        sched.scheduler_step(true, true);



        auto pmass = setup.get_part_mass();

        auto Npart = 8./pmass;

        //setup.pertub_eigenmode_wave(sched, {0,0}, {0,0,1}, 0);

        for(u32 i = 0; i < 5; i++){
            setup.update_smoothing_lenght(sched);
        }

        return {pmass,Npart};
    };

    auto [pmass,Npart] = setup();


    


    if(sched.patch_list.global.size() != npatch){
        throw ShamrockSyclException("Wrong patch count" + format("%d, wanted %d",sched.patch_list.global.size(),npatch));
    }

    

    using Model = models::sph::BasicSPHGas<f32, models::sph::kernels::M4<f32>>;

    Model model ;

    const f32 htol_up_tol  = 1.4;
    const f32 htol_up_iter = 1.2;

    const f32 cfl_cour  = 0.02;
    const f32 cfl_force = 0.3;

    model.set_cfl_force(0.3);
    model.set_cfl_cour(0.25);
    model.set_particle_mass(pmass);

    shamsys::instance::get_compute_queue().wait();

    Timer t;
    t.start();
    
    f64 t2 = model.evolve(sched, 0, 100);
    shamsys::instance::get_compute_queue().wait();

    t.end();

    model.close();

    return {Npart,t.nanosec/1.e9, 2./t2};

}

template<class flt>
void benchmark_periodic_box_main(u32 npatch, std::string name){


    std::vector<f64> npart;
    std::vector<f64> times;
    std::vector<f64> niter;

    {

        
        f64 part_per_g = 4000000;

        f64 gsz = shamsys::instance::get_compute_queue().get_device().get_info<sycl::info::device::global_mem_size>();
        gsz = 1024*1024*1024*1;

        logger::raw_ln("limit = ", part_per_g*(gsz/1.3)/(1024.*1024.*1024.));
    }

    auto should_stop = [&](f64 dr){

        f64 part_per_g = 4000000;

        f64 Nesti = (1.F/dr)*(1.F/dr)*(1.F/dr);

        f64 multiplier = shamsys::instance::world_size;

        if(npatch < multiplier){
            multiplier = 1;
        }

        f64 gsz = shamsys::instance::get_compute_queue().get_device().get_info<sycl::info::device::global_mem_size>();
        gsz = 1024*1024*1024*1;

        f64 a = (Nesti/part_per_g)*1024.*1024.*1024.;
        f64 b = multiplier*gsz/1.3;


        logger::raw_ln(Nesti,a,b);

        return a < b;

    };



    f32 dr = 0.05;
    for(; should_stop(dr); dr /= 1.1){
        auto [N,t, ni] = benchmark_periodic_box<flt>(dr, npatch);
        npart.push_back(N);
        times.push_back(t);
        niter.push_back(ni);
    }

    if(shamsys::instance::world_rank == 0){
        auto & dset = shamtest::test_data().new_dataset(name);
        dset.add_data("Npart", npart);
        dset.add_data("times", times);
        dset.add_data("niter", niter);
    }

}

TestStart(Benchmark, "benchmark periodic box sph", bench_per_box_sph, -1){

    benchmark_periodic_box_main<f32>(1,"patch_1");
    benchmark_periodic_box_main<f32>(8,"patch_8");
    benchmark_periodic_box_main<f32>(64,"patch_64");

}
