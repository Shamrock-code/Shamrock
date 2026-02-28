// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file benchmark_selfgrav.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/time.hpp"
#include "shammodels/nbody/models/nbody_selfgrav.hpp"
#include "shammodels/nbody/setup/nbody_setup.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtest/shamtest.hpp"
#include <string>

template<class flt>
std::tuple<f64, f64> benchmark_selfgrav(f32 dr, u32 npatch) {

    using namespace shamrock::patch;

    using vec = sycl::vec<flt, 3>;

    u64 Nesti = (2.F / dr) * (2.F / dr) * (2.F / dr);

    std::shared_ptr<shamrock::patch::PatchDataLayout> pdl_ptr
        = std::make_shared<shamrock::patch::PatchDataLayout>(std::vector<std::string>{"main"});
    auto &pdl = pdl_ptr->get_layer_ref(0);

    pdl.add_field<f32_3>("xyz", 1);
    pdl.add_field<f32>("hpart", 1);
    pdl.add_field<f32_3>("vxyz", 1);
    pdl.add_field<f32_3>("axyz", 1);
    pdl.add_field<f32_3>("axyz_old", 1);

    auto id_v = pdl.get_field_idx<f32_3>("vxyz");
    auto id_a = pdl.get_field_idx<f32_3>("axyz");

    PatchScheduler sched = PatchScheduler(pdl_ptr, Nesti / npatch, 1);
    sched.init_mpi_required_types();

    auto setup = [&]() -> std::tuple<flt, f64> {
        using Setup = models::nbody::NBodySetup<f32>;

        Setup setup;
        setup.init(sched);

        auto box = setup.get_ideal_box(dr, {vec{-1, -1, -1}, vec{1, 1, 1}});

        // auto ebox = box;
        // std::get<0>(ebox).x() -= 1e-5;
        // std::get<0>(ebox).y() -= 1e-5;
        // std::get<0>(ebox).z() -= 1e-5;
        // std::get<1>(ebox).x() += 1e-5;
        // std::get<1>(ebox).y() += 1e-5;
        // std::get<1>(ebox).z() += 1e-5;
        sched.set_coord_domain_bound<f32_3>(box);

        setup.set_boundaries(true);
        setup.add_particules_fcc(sched, dr, box);
        setup.set_total_mass(8.);

        sched.for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
            pdat.get_field<f32_3>(id_v).override(f32_3{0, 0, 0});
            pdat.get_field<f32_3>(id_a).override(f32_3{0, 0, 0});
        });

        sched.scheduler_step(true, true);

        sched.scheduler_step(true, true);

        sched.scheduler_step(true, true);

        auto pmass = setup.get_part_mass();

        auto Npart = 8. / pmass;

        // setup.pertub_eigenmode_wave(sched, {0,0}, {0,0,1}, 0);

        return {pmass, Npart};
    };

    auto [pmass, Npart] = setup();

    if (sched.patch_list.global.size() != npatch) {
        throw ShamrockSyclException(
            "Wrong patch count"
            + shambase::format_printf("%d, wanted %d", sched.patch_list.global.size(), npatch));
    }

    using Model = models::nbody::Nbody_SelfGrav<f32>;

    Model model;

    const f32 htol_up_tol  = 1.4;
    const f32 htol_up_iter = 1.2;

    const f32 cfl_cour  = 0.02;
    const f32 cfl_force = 0.3;

    model.set_cfl_force(0.3);
    model.set_particle_mass(pmass);

    shamsys::instance::get_compute_queue().wait();

    shambase::Timer t;
    t.start();

    model.evolve(sched, 0, 1e-3);
    shamsys::instance::get_compute_queue().wait();

    t.end();

    model.close();

    return {Npart, t.nanosec / 1e9};
}

template<class flt>
void benchmark_selfgrav_main(u32 npatch, std::string name) {

    std::vector<f64> npart;
    std::vector<f64> times;

    {

        f64 part_per_g = 2500000;

        f64 gsz = shamsys::instance::get_compute_queue()
                      .get_device()
                      .get_info<sycl::info::device::global_mem_size>();
        gsz = 1024 * 1024 * 1024 * 1;

        logger::raw_ln("limit = ", part_per_g * (gsz / 1.3) / (1024. * 1024. * 1024.));
    }

    auto should_stop = [&](f64 dr) {
        f64 part_per_g = 2500000;

        f64 Nesti = (1.F / dr) * (1.F / dr) * (1.F / dr);

        f64 multiplier = shamcomm::world_size();

        if (npatch < multiplier) {
            multiplier = 1;
        }

        f64 gsz = shamsys::instance::get_compute_queue()
                      .get_device()
                      .get_info<sycl::info::device::global_mem_size>();
        gsz = 1024 * 1024 * 1024 * 1;

        f64 a = (Nesti / part_per_g) * 1024. * 1024. * 1024.;
        f64 b = multiplier * gsz / 1.3;

        logger::raw_ln(Nesti, a, b);

        return a < b;
    };

    f32 dr = 0.05;
    for (; should_stop(dr); dr /= 1.1) {
        auto [N, t] = benchmark_selfgrav<flt>(dr, npatch);
        npart.push_back(N);
        times.push_back(t);
    }

    if (shamcomm::world_rank() == 0) {
        auto &dset = shamtest::test_data().new_dataset(name);
        dset.add_data("Npart", npart);
        dset.add_data("times", times);
    }
}

TestStart(Benchmark, "benchmark selfgrav nbody", bench_selfgrav_nbody, -1) {

    benchmark_selfgrav_main<f32>(1, "patch_1");
    // benchmark_selfgrav_main<f32>(8,"patch_8");
    // benchmark_selfgrav_main<f32>(64,"patch_64");
}
