// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <memory>
#include <vector>

template<class Tvec, class Tscal, class SPHKernel>
void test_smoothing_length_density_module(
    std::vector<Tvec> &positions_vec,
    std::vector<Tscal> &start_h_vec,

    std::vector<Tscal> &expected_h_vec_end,
    std::vector<Tscal> &expected_eps_vec_end,
    std::vector<Tscal> &expected_sequence_eps_min,
    std::vector<Tscal> &expected_sequence_eps_max,
    std::vector<Tscal> &expected_sequence_h_min,
    std::vector<Tscal> &expected_sequence_h_max,

    Tscal gpart_mass,
    Tscal h_evol_max,
    Tscal h_evol_iter_max) {
    using namespace shamrock;
    using namespace shammodels::sph::modules;
    using namespace shammodels::sph::solvergraph;

    u32 N_particles = positions_vec.size();

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

    // move it into a INode ptr to test polymorphism
    std::shared_ptr<solvergraph::INode> iterate_module_ptr
        = std::make_shared<decltype(iterate_module)>(std::move(iterate_module));

    std::vector<Tscal> eps_min_sequence;
    std::vector<Tscal> eps_max_sequence;
    std::vector<Tscal> h_min_sequence;
    std::vector<Tscal> h_max_sequence;

    // 9. Run the module
    for (u32 outer_iter = 0; outer_iter < 50; ++outer_iter) {

        //  Overwrite the old h values with the new h values
        old_h_field.get_buf().copy_from_stdvec(new_h_field.get_buf().copy_to_stdvec());
        eps_h_field.get_buf().fill(10000000.);

        Tscal max_eps_h = 10000000.;
        for (u32 inner_iter = 0; inner_iter < 10; ++inner_iter) {

            // Run the module
            iterate_module_ptr->evaluate();

            // Get results back from device
            std::vector<Tscal> new_h_result = new_h_field.get_buf().copy_to_stdvec();
            std::vector<Tscal> eps_h_result = eps_h_field.get_buf().copy_to_stdvec();

            // print h_max and h_min
            // shamcomm::logs::raw_ln(
            //     "h_max", *std::max_element(new_h_result.begin(), new_h_result.end()));
            // shamcomm::logs::raw_ln(
            //     "h_min", *std::min_element(new_h_result.begin(), new_h_result.end()));

            // add results to sequences for validation
            {
                eps_min_sequence.push_back(eps_h_field.compute_min());
                eps_max_sequence.push_back(eps_h_field.compute_max());
                h_min_sequence.push_back(
                    *std::min_element(new_h_result.begin(), new_h_result.end()));
                h_max_sequence.push_back(
                    *std::max_element(new_h_result.begin(), new_h_result.end()));
            }

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

            // get max eps_h
            max_eps_h = eps_h_field.compute_max();
            // shamcomm::logs::raw_ln("max_eps_h", max_eps_h);

            // either converged or need a new outer iteration (early break)
            if (max_eps_h < 1e-6) {
                break;
            }
        }

        // get min eps_h
        Tscal min_eps_h = eps_h_field.compute_min();
        // shamcomm::logs::raw_ln("min_eps_h", min_eps_h);

        // need a new outer iteration
        if (min_eps_h == -1) {
            continue;
        }

        // is converged, we can break
        if (max_eps_h < 1e-6) {
            break;
        }

        // if last outer iteration, we need to crash, since we did not converge
        if (outer_iter == 49) {
            REQUIRE_NAMED("The h iterator is not converged after 50 iterations", false);
        }
    }

    // 10. Compare the results with the expected values
    {
        std::vector<Tscal> new_h_result = new_h_field.get_buf().copy_to_stdvec();
        std::vector<Tscal> eps_h_result = eps_h_field.get_buf().copy_to_stdvec();

        constexpr Tscal tol = 1e-6;
        auto almost_equal   = [tol](const std::vector<Tscal> &a, const std::vector<Tscal> &b) {
            // check sizes
            if (a.size() != b.size()) {
                return false;
            }

            // check values
            for (u32 i = 0; i < a.size(); ++i) {
                if (std::abs(a[i] - b[i]) > tol) {
                    return false;
                }
            }
            return true;
        };

        REQUIRE_EQUAL_CUSTOM_COMP(new_h_result, expected_h_vec_end, almost_equal);
        REQUIRE_EQUAL_CUSTOM_COMP(eps_h_result, expected_eps_vec_end, almost_equal);
        REQUIRE_EQUAL_CUSTOM_COMP(eps_min_sequence, expected_sequence_eps_min, almost_equal);
        REQUIRE_EQUAL_CUSTOM_COMP(eps_max_sequence, expected_sequence_eps_max, almost_equal);
        REQUIRE_EQUAL_CUSTOM_COMP(h_min_sequence, expected_sequence_h_min, almost_equal);
        REQUIRE_EQUAL_CUSTOM_COMP(h_max_sequence, expected_sequence_h_max, almost_equal);
    }
}

TestStart(
    Unittest,
    "shammodels/sph/modules/IterateSmoothingLengthDensity",
    IterateSmoothingLengthDensity_base,
    1) {
    using Tvec      = f64_3;
    using Tscal     = f64;
    using SPHKernel = shammath::M4<f64>;

    constexpr Tscal gpart_mass      = 1.0;
    constexpr Tscal h_evol_max      = 1.2;
    constexpr Tscal h_evol_iter_max = 1.2;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Parameters of the test
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Tvec> positions_vec;

    for (u32 i = 0; i < 4; ++i) {
        for (u32 j = 0; j < 4; ++j) {
            for (u32 k = 0; k < 4; ++k) {
                Tvec pos = {(Tscal) i, (Tscal) j, (Tscal) k};
                positions_vec.push_back(pos);
            }
        }
    }

    u32 N_particles = positions_vec.size();

    std::vector<Tscal> start_h_vec(N_particles, 0.1);

    std::vector<Tscal> expected_h_vec_end{
        1.7039887744498596, 1.4852333966670939, 1.4852333966670939, 1.7039887744498596,
        1.4852333966670939, 1.3212315623706217, 1.3212315623706217, 1.4852333966670936,
        1.4852333966670939, 1.3212315623706217, 1.321231562370622,  1.4852333966670936,
        1.7039887744498596, 1.485233396667094,  1.485233396667094,  1.7039887744498594,
        1.4852333966670939, 1.3212315623706217, 1.3212315623706217, 1.485233396667094,
        1.3212315623706217, 1.2024373044416155, 1.2024373044416155, 1.3212315623706217,
        1.3212315623706217, 1.2024373044416155, 1.2024373044416155, 1.3212315623706217,
        1.485233396667094,  1.321231562370622,  1.3212315623706217, 1.4852333966670939,
        1.4852333966670943, 1.3212315623706214, 1.321231562370622,  1.4852333966670943,
        1.3212315623706217, 1.2024373044416155, 1.2024373044416155, 1.321231562370622,
        1.3212315623706217, 1.2024373044416155, 1.2024373044416155, 1.321231562370622,
        1.4852333966670943, 1.321231562370622,  1.321231562370622,  1.4852333966670945,
        1.7039887744498599, 1.485233396667094,  1.485233396667094,  1.7039887744498599,
        1.485233396667094,  1.321231562370622,  1.3212315623706217, 1.485233396667094,
        1.485233396667094,  1.321231562370622,  1.321231562370622,  1.485233396667094,
        1.7039887744498599, 1.4852333966670945, 1.4852333966670945, 1.70398877444986};

    std::vector<Tscal> expected_eps_vec_end{
        6.3426814945658865e-12, 5.980059576449213e-16,  4.48504468233691e-16,
        6.34224913727246e-12,   4.48504468233691e-16,   5.041764318586834e-16,
        5.041764318586834e-16,  4.485044682336911e-16,  4.48504468233691e-16,
        5.041764318586834e-16,  5.041764318586834e-16,  4.485044682336911e-16,
        6.34224913727246e-12,   1.4950148941123038e-16, 1.4950148941123038e-16,
        6.342105018174651e-12,  4.48504468233691e-16,   5.041764318586834e-16,
        5.041764318586834e-16,  2.990029788224607e-16,  5.041764318586834e-16,
        0.0000000000000000,     0.0000000000000000,     5.041764318586834e-16,
        3.3611762123912236e-16, 0.0000000000000000,     0.0000000000000000,
        3.3611762123912236e-16, 1.4950148941123038e-16, 0.0000000000000000,
        3.3611762123912236e-16, 2.9900297882246075e-16, 1.4950148941123035e-16,
        5.041764318586835e-16,  0.0000000000000000,     1.4950148941123035e-16,
        3.3611762123912236e-16, 0.0000000000000000,     0.0000000000000000,
        0.0000000000000000,     3.3611762123912236e-16, 0.0000000000000000,
        0.0000000000000000,     0.0000000000000000,     0.0000000000000000,
        0.0000000000000000,     1.6805881061956118e-16, 0.0000000000000000,
        6.342537375468078e-12,  1.4950148941123038e-16, 1.4950148941123038e-16,
        6.342537375468078e-12,  1.4950148941123038e-16, 1.6805881061956118e-16,
        3.3611762123912236e-16, 1.4950148941123038e-16, 1.4950148941123038e-16,
        1.6805881061956123e-16, 0.0000000000000000,     1.4950148941123038e-16,
        6.342537375468078e-12,  0.0000000000000000,     0.0000000000000000,
        6.3426814945658865e-12};

    std::vector<Tscal> expected_sequence_eps_min{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1, 0,  -1,
                                                 -1, -1, -1, 0,  0,  0,  0,  0};

    std::vector<Tscal> expected_sequence_eps_max{
        -1.0000000000000000,    -1.0000000000000000,    -1.0000000000000000,
        -1.0000000000000000,    -1.0000000000000000,    -1.0000000000000000,
        -1.0000000000000000,    -1.0000000000000000,    -1.0000000000000000,
        -1.0000000000000000,    -1.0000000000000000,    -1.0000000000000000,
        -1.0000000000000000,    0.19667779953573308,    0.026408786682080153,
        0.0015808702946168102,  5.25907353009218e-06,   5.793905946882268e-11,
        0.18226586196218933,    0.03892212639115773,    0.0035984720083355017,
        2.8023453743763244e-05, 1.6838139227630172e-09, 0.08452448630589766,
        0.020529594483443016,   0.0009261073956681399,  1.7604940220352781e-06,
        6.3426814945658865e-12};

    std::vector<Tscal> expected_sequence_h_min{
        0.12000000000000000, 0.14400000000000000, 0.17279999999999998, 0.20735999999999996,
        0.24883199999999994, 0.29859839999999993, 0.3583180799999999,  0.4299816959999999,
        0.5159780351999999,  0.6191736422399998,  0.7430083706879997,  0.8916100448255997,
        1.0699320537907195,  1.1724846463543073,  1.200740253727186,   1.202431677528282,
        1.2024373043796248,  1.2024373044416155,  1.2024373044416155,  1.2024373044416155,
        1.2024373044416155,  1.2024373044416155,  1.2024373044416155,  1.2024373044416155,
        1.2024373044416155,  1.2024373044416155,  1.2024373044416155,  1.2024373044416155};

    std::vector<Tscal> expected_sequence_h_max{
        0.12000000000000000, 0.14400000000000000, 0.17279999999999998, 0.20735999999999996,
        0.24883199999999994, 0.29859839999999993, 0.3583180799999999,  0.4299816959999999,
        0.5159780351999999,  0.6191736422399998,  0.7430083706879997,  0.8916100448255997,
        1.0699320537907195,  1.2839184645488635,  1.2839184645488635,  1.2839184645488635,
        1.2839184645488635,  1.2839184645488635,  1.5179329701790327,  1.5407021574586361,
        1.5407021574586361,  1.5407021574586361,  1.5407021574586361,  1.6709292158682156,
        1.702559206380607,   1.7039860620431506,  1.7039887744400881,  1.70398877444986};

    test_smoothing_length_density_module<Tvec, Tscal, SPHKernel>(
        positions_vec,
        start_h_vec,
        expected_h_vec_end,
        expected_eps_vec_end,
        expected_sequence_eps_min,
        expected_sequence_eps_max,
        expected_sequence_h_min,
        expected_sequence_h_max,
        gpart_mass,
        h_evol_max,
        h_evol_iter_max);
}
