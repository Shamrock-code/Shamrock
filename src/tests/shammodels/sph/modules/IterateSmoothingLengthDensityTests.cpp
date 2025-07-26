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

    constexpr u32 N_particles       = 4 * 4 * 4;
    constexpr Tscal gpart_mass      = 1.0;
    constexpr Tscal h_evol_max      = 1.1;
    constexpr Tscal h_evol_iter_max = 1.1;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Parameters of the test
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Tvec> positions_vec;
    positions_vec.reserve(N_particles);

    // Create particles in a 8x8x8 cube pattern
    for (u32 i = 0; i < 4; ++i) {
        for (u32 j = 0; j < 4; ++j) {
            for (u32 k = 0; k < 4; ++k) {
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
    auto &q        = dev_sched->get_queue();

    // 2. Create initial smoothing lengths (all particles have h = 1.0)
    std::vector<Tscal> old_h_vec(start_h_vec);
    std::vector<Tscal> new_h_vec(start_h_vec);
    std::vector<Tscal> eps_h_vec(N_particles, 10000000.); // Start with non-zero epsilon

    // 3. Create mock neighbor cache where everyone is neighbor of everyone
    // This creates a fully connected graph for testing

    sham::DeviceBuffer<u32> neigh_count(N_particles, dev_sched);
    neigh_count.fill(N_particles);

    tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), N_particles);

    sham::kernel_call(
        q,
        sham::MultiRef{pcache.scanned_cnt},
        sham::MultiRef{pcache.index_neigh_map},
        N_particles,
        [Npart
         = N_particles](u32 id_a, const u32 *__restrict scanned_neigh_cnt, u32 *__restrict neigh) {
            u32 cnt = scanned_neigh_cnt[id_a];

            for (u32 i = 0; i < Npart; ++i) {
                neigh[cnt] = i;
                cnt += 1;
            }
        });

    // for debugging
    // shamcomm::logs::raw_ln("pcache.index_neigh_map", pcache.index_neigh_map.copy_to_stdvec());

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
    sizes->indexes.add_obj(0, u32{N_particles});

    auto positions_refs = std::make_shared<solvergraph::FieldRefs<Tvec>>("positions", "positions");
    auto positions_refs_data = shamrock::solvergraph::DDPatchDataFieldRef<Tvec>{};
    positions_refs_data.add_obj(0, std::ref(positions_field));
    positions_refs->set_refs(positions_refs_data);

    auto old_h_refs      = std::make_shared<solvergraph::FieldRefs<Tscal>>("old_h", "old_h");
    auto old_h_refs_data = shamrock::solvergraph::DDPatchDataFieldRef<Tscal>{};
    old_h_refs_data.add_obj(0, std::ref(old_h_field));
    old_h_refs->set_refs(old_h_refs_data);

    auto new_h_refs      = std::make_shared<solvergraph::FieldRefs<Tscal>>("new_h", "new_h");
    auto new_h_refs_data = shamrock::solvergraph::DDPatchDataFieldRef<Tscal>{};
    new_h_refs_data.add_obj(0, std::ref(new_h_field));
    new_h_refs->set_refs(new_h_refs_data);

    auto eps_h_refs      = std::make_shared<solvergraph::FieldRefs<Tscal>>("eps_h", "eps_h");
    auto eps_h_refs_data = shamrock::solvergraph::DDPatchDataFieldRef<Tscal>{};
    eps_h_refs_data.add_obj(0, std::ref(eps_h_field));
    eps_h_refs->set_refs(eps_h_refs_data);

    auto neigh_cache = std::make_shared<NeighCache>("neigh_cache", "neigh_cache");
    neigh_cache->neigh_cache.add_obj(0, std::move(pcache));

    // 8. Set up IterateSmoothingLengthDensity module
    IterateSmoothingLengthDensity<Tvec, SPHKernel> iterate_module(
        gpart_mass, h_evol_max, h_evol_iter_max);
    iterate_module.set_edges(
        sizes, neigh_cache, positions_refs, old_h_refs, new_h_refs, eps_h_refs);

    // 9. Run the module
    for (u32 outer_iter = 0; outer_iter < 10; ++outer_iter) {

        //  Overwrite the old h values with the new h values
        old_h_field.get_buf().copy_from_stdvec(new_h_field.get_buf().copy_to_stdvec());

        for (u32 inner_iter = 0; inner_iter < 10; ++inner_iter) {

            // Run the module
            iterate_module.evaluate();

            // Get results back from device
            std::vector<Tscal> new_h_result = new_h_field.get_buf().copy_to_stdvec();
            std::vector<Tscal> eps_h_result = eps_h_field.get_buf().copy_to_stdvec();

            // print h_max and h_min
            shamcomm::logs::raw_ln(
                "h_max", *std::max_element(new_h_result.begin(), new_h_result.end()));
            shamcomm::logs::raw_ln(
                "h_min", *std::min_element(new_h_result.begin(), new_h_result.end()));

            // Verify results
            u32 eps_expected_range_offenses = 0;
            for (u32 i = 0; i < N_particles; ++i) {
                if (!(eps_h_result[i] >= 0.0 || eps_h_result[i] == -1.0)) {
                    eps_expected_range_offenses += 1;
                }
            }
            REQUIRE_EQUAL(eps_expected_range_offenses, 0);

            u32 new_h_expected_range_offenses = 0;
            for (u32 i = 0; i < N_particles; ++i) {
                if (!(new_h_result[i] > 0.0 && new_h_result[i] <= 10.0)) {
                    new_h_expected_range_offenses += 1;
                }
            }
            REQUIRE_EQUAL(new_h_expected_range_offenses, 0);
        }
    }

    // 10. Compare the results with the expected values
    {
        std::vector<Tscal> new_h_result = new_h_field.get_buf().copy_to_stdvec();
        std::vector<Tscal> eps_h_result = eps_h_field.get_buf().copy_to_stdvec();

        REQUIRE_EQUAL(new_h_result, expected_h_vec);
        REQUIRE_EQUAL(eps_h_result, expected_eps_vec);
    }
}
