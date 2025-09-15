// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeNeighStats.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the ComputeNeighStats module for analyzing neighbor counts.
 *
 */

#include "shammodels/sph/modules/ComputeNeighStats.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    void ComputeNeighStats<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        const shambase::DistributedData<u32> &counts = edges.part_counts.indexes;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal Rker2 = kernel_radius * kernel_radius;

        counts.for_each([&](u64 id, u32 count) {
            const shamrock::tree::ObjectCache &pcache = edges.neigh_cache.get_cache(id);

            sham::DeviceBuffer<u32> neigh_count_all(
                count, shamsys::instance::get_compute_scheduler_ptr());
            sham::DeviceBuffer<u32> neigh_count_true(
                count, shamsys::instance::get_compute_scheduler_ptr());

            sham::kernel_call(
                q,
                sham::MultiRef{
                    pcache, edges.xyz.get_spans().get(id), edges.hpart.get_spans().get(id)},
                sham::MultiRef{neigh_count_true, neigh_count_all},
                count,
                [&, Rker2](
                    u32 id_a,
                    const auto ploop_ptrs,
                    const Tvec *xyz,
                    const Tscal *hpart,
                    u32 *neigh_count_true,
                    u32 *neigh_count_all) {
                    shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                    Tscal h_a  = hpart[id_a];
                    Tvec xyz_a = xyz[id_a];

                    u32 cnt_all  = 0;
                    u32 cnt_true = 0;
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        cnt_all++;

                        Tvec r_ab  = xyz_a - xyz[id_b];
                        Tscal rab2 = sycl::dot(r_ab, r_ab);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        cnt_true++;
                    });

                    neigh_count_all[id_a]  = cnt_all;
                    neigh_count_true[id_a] = cnt_true;
                });

            {
                std::vector<u32> neigh_cnt_vec = neigh_count_true.copy_to_stdvec();

                double max  = *std::max_element(neigh_cnt_vec.begin(), neigh_cnt_vec.end());
                double min  = *std::min_element(neigh_cnt_vec.begin(), neigh_cnt_vec.end());
                double mean = std::accumulate(neigh_cnt_vec.begin(), neigh_cnt_vec.end(), 0.0)
                              / neigh_cnt_vec.size();
                double stddev = std::sqrt(
                    std::accumulate(
                        neigh_cnt_vec.begin(),
                        neigh_cnt_vec.end(),
                        0.0,
                        [mean](double acc, double x) {
                            return acc + (x - mean) * (x - mean);
                        })
                    / neigh_cnt_vec.size());
                logger::raw_ln("(true) max, min, mean, stddev =", max, min, mean, stddev);
            }

            {
                std::vector<u32> neigh_cnt_vec = neigh_count_all.copy_to_stdvec();

                double max  = *std::max_element(neigh_cnt_vec.begin(), neigh_cnt_vec.end());
                double min  = *std::min_element(neigh_cnt_vec.begin(), neigh_cnt_vec.end());
                double mean = std::accumulate(neigh_cnt_vec.begin(), neigh_cnt_vec.end(), 0.0)
                              / neigh_cnt_vec.size();
                double stddev = std::sqrt(
                    std::accumulate(
                        neigh_cnt_vec.begin(),
                        neigh_cnt_vec.end(),
                        0.0,
                        [mean](double acc, double x) {
                            return acc + (x - mean) * (x - mean);
                        })
                    / neigh_cnt_vec.size());
                logger::raw_ln("(all) max, min, mean, stddev =", max, min, mean, stddev);
            }
        });
    }

    template<class Tvec>
    std::string ComputeNeighStats<Tvec>::_impl_get_tex() {
        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::ComputeNeighStats<f64_3>;
