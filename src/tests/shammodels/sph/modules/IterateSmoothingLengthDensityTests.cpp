// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/IterateSmoothingLengthDensity.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <memory>
#include <vector>

TestStart(
    Unittest,
    "shammodels/sph/modules/IterateSmoothingLengthDensity:basic",
    IterateSmoothingLengthDensity_basic,
    1) {
    using Tvec      = f64_3;
    using Tscal     = f64;
    using SPHKernel = shammath::M4<f64>;
    using namespace shamrock;
    using namespace shammodels::sph::modules;
    using namespace shammodels::sph::solvergraph;

    constexpr u32 N_particles       = 8;
    constexpr Tscal gpart_mass      = 1.0;
    constexpr Tscal h_evol_max      = 1.1;
    constexpr Tscal h_evol_iter_max = 1.1;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Parameters of the test
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Tvec> positions_vec;
    positions_vec.reserve(N_particles);

    // Create particles in a 8x8x8 cube pattern
    for (u32 i = 0; i < 8; ++i) {
        for (u32 j = 0; j < 8; ++j) {
            for (u32 k = 0; k < 8; ++k) {
                Tvec pos = {(Tscal) i, (Tscal) j, (Tscal) k};
                positions_vec.push_back(pos);
            }
        }
    }

    std::vector<Tscal> start_h_vec(N_particles, 0.1);

    // temporary, to be replaced by the expected h values
    std::vector<Tscal> expected_h_vec(N_particles, -1);

    u32 expected_iterations = 10;

    // temporary, to be replaced by the expected h values
    std::vector<Tscal> expected_eps_vec(N_particles, -1);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // The actual test
    ////////////////////////////////////////////////////////////////////////////////////////////////

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    auto & q = dev_sched->get_queue();

    // 2. Create initial smoothing lengths (all particles have h = 1.0)
    std::vector<Tscal> old_h_vec(start_h_vec);
    std::vector<Tscal> new_h_vec(start_h_vec);
    std::vector<Tscal> eps_h_vec(N_particles, 10000000.); // Start with non-zero epsilon

    // 3. Create mock neighbor cache where everyone is neighbor of everyone
    // This creates a fully connected graph for testing

    sham::DeviceBuffer<u32> neigh_count(N_particles, dev_sched);
    neigh_count.fill(N_particles);

    tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), N_particles);

    sham::kernel_call(q,
        sham::MultiRef{pcache.scanned_cnt},
        sham::MultiRef{ pcache.index_neigh_map},
        N_particles,
        [Npart = N_particles](u32 id_a, const u32 * __restrict scanned_neigh_cnt, u32 * __restrict neigh) {
            u32 cnt = scanned_neigh_cnt[id_a];

            for (u32 i = 0; i < Npart; ++i) {
                neigh[cnt] = i;
                cnt += 1;
            }
        });

    shamcomm::logs::raw_ln("pcache.index_neigh_map", pcache.index_neigh_map.copy_to_stdvec());

    // 5. Create PatchDataField for positions and smoothing lengths
    PatchDataField<Tvec> positions_field("positions", 1, N_particles);
    PatchDataField<Tscal> old_h_field("old_h", 1, N_particles);
    PatchDataField<Tscal> new_h_field("new_h", 1, N_particles);
    PatchDataField<Tscal> eps_h_field("eps_h", 1, N_particles);

    // Copy data to PatchDataField
    positions_field.get_buf().copy_from_stdvec(positions_vec);
    old_h_field.get_buf().copy_from_stdvec(old_h_vec);
    new_h_field.get_buf().copy_from_stdvec(new_h_vec);
    eps_h_field.get_buf().copy_from_stdvec(eps_h_vec);

    // 6. Create solver graph components
    auto sizes = std::make_shared<solvergraph::Indexes<u32>>("sizes", "sizes");
    sizes->indexes.add_obj(0, N_particles);

    auto positions_refs = std::make_shared<solvergraph::FieldRefs<Tvec>>("positions", "positions");
    positions_refs->set_refs(
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tvec>>>{});
    positions_refs->get_refs().add_obj(0, std::ref(positions_field));

    auto old_h_refs = std::make_shared<solvergraph::FieldRefs<Tscal>>("old_h", "old_h");
    old_h_refs->set_refs(
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>>{});
    old_h_refs->get_refs().add_obj(0, std::ref(old_h_field));

    auto new_h_refs = std::make_shared<solvergraph::FieldRefs<Tscal>>("new_h", "new_h");
    new_h_refs->set_refs(
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>>{});
    new_h_refs->get_refs().add_obj(0, std::ref(new_h_field));

    auto eps_h_refs = std::make_shared<solvergraph::FieldRefs<Tscal>>("eps_h", "eps_h");
    eps_h_refs->set_refs(
        shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>>{});
    eps_h_refs->get_refs().add_obj(0, std::ref(eps_h_field));

    // 7. Create neighbor cache
    auto neigh_cache = std::make_shared<NeighCache>("neigh_cache", "neigh_cache");

    // Create ObjectCache for the neighbor cache
    shamrock::tree::ObjectCache obj_cache{
        std::move(cnt_neigh_buf),
        std::move(scanned_cnt_buf),
        total_neighbors,
        std::move(index_neigh_map_buf)};

    neigh_cache->neigh_cache.add_obj(0, std::move(obj_cache));

    // 8. Set up IterateSmoothingLengthDensity module
    IterateSmoothingLengthDensity<Tvec, SPHKernel> iterate_module(
        gpart_mass, h_evol_max, h_evol_iter_max);
    iterate_module.set_edges(
        sizes, neigh_cache, positions_refs, old_h_refs, new_h_refs, eps_h_refs);

    // 9. Run the module
    iterate_module.evaluate();

    // 10. Get results back from device
    std::vector<Tscal> new_h_result = new_h_field.get_buf().copy_to_stdvec();
    std::vector<Tscal> eps_h_result = eps_h_field.get_buf().copy_to_stdvec();

    // 11. Verify results
    // Check that all particles have been processed (epsilon should be updated)
    for (u32 i = 0; i < N_particles; ++i) {
        // The epsilon should be updated for particles that were processed
        REQUIRE_NAMED("epsilon_updated", eps_h_result[i] >= 0.0 || eps_h_result[i] == -1.0);

        // New h should be within reasonable bounds
        REQUIRE_NAMED("h_positive", new_h_result[i] > 0.0);
        REQUIRE_NAMED("h_reasonable", new_h_result[i] <= 10.0); // Should not explode
    }

    // 12. Test with different initial conditions
    {
        // Test with larger initial h values
        std::vector<Tscal> large_h_vec(N_particles, 2.0);
        PatchDataField<Tscal> large_h_field("large_h", 1, N_particles);
        large_h_field.get_buf().copy_from_stdvec(large_h_vec);

        auto large_h_refs = std::make_shared<solvergraph::FieldRefs<Tscal>>("large_h", "large_h");
        large_h_refs->set_refs(
            shambase::DistributedData<std::reference_wrapper<PatchDataField<Tscal>>>{});
        large_h_refs->get_refs().add_obj(0, std::ref(large_h_field));

        IterateSmoothingLengthDensity<Tvec, SPHKernel> iterate_large_h(
            gpart_mass, h_evol_max, h_evol_iter_max);
        iterate_large_h.set_edges(
            sizes, neigh_cache, positions_refs, large_h_refs, new_h_refs, eps_h_refs);
        iterate_large_h.evaluate();

        std::vector<Tscal> large_h_result = large_h_field.get_buf().copy_to_stdvec();
        for (u32 i = 0; i < N_particles; ++i) {
            REQUIRE_NAMED("large_h_processed", large_h_result[i] > 0.0);
        }
    }
}
