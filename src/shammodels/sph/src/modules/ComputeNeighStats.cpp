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

#include "shambase/DistributedData.hpp"
#include "nlohmann/json.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/sph/modules/ComputeNeighStats.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include <vector>

namespace {

    bool neigh_stats_json_loaded = false;
    nlohmann::json neigh_stats_json;
    std::string neigh_stats_json_filename;

    void load_neigh_stats_json(const std::string &filename) {
        neigh_stats_json_filename = filename;

        // load it if it exists
        if (std::ifstream(filename).good()) {
            neigh_stats_json = nlohmann::json::parse(std::ifstream(filename));
        } else {
            neigh_stats_json = nlohmann::json::array();
        }
    }

    void save_neigh_stats_json() {
        std::ofstream file(neigh_stats_json_filename);
        file << neigh_stats_json.dump(4) << std::endl;
    }

    void register_entry(
        f64 time,
        f64 max_true,
        f64 min_true,
        f64 mean_true,
        f64 stddev_true,
        f64 max_all,
        f64 min_all,
        f64 mean_all,
        f64 stddev_all) {
        // neigh stats json is a list so we append this as a new entry
        neigh_stats_json.push_back(
            {{"time", time},
             {"max_true", max_true},
             {"min_true", min_true},
             {"mean_true", mean_true},
             {"stddev_true", stddev_true},
             {"max_all", max_all},
             {"min_all", min_all},
             {"mean_all", mean_all},
             {"stddev_all", stddev_all}});
        save_neigh_stats_json();
    }

} // namespace

namespace shammodels::sph::modules {

    template<class Tvec>
    void ComputeNeighStats<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        const shambase::DistributedData<u32> &counts = edges.part_counts.indexes;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal Rker2 = kernel_radius * kernel_radius;

        shambase::DistributedData<sham::DeviceBuffer<u32>> neigh_count_all;
        shambase::DistributedData<sham::DeviceBuffer<u32>> neigh_count_true;

        counts.for_each([&](u64 id, u32 count) {
            const shamrock::tree::ObjectCache &pcache = edges.neigh_cache.get_cache(id);

            neigh_count_all.add_obj(
                id, sham::DeviceBuffer<u32>(count, shamsys::instance::get_compute_scheduler_ptr()));
            neigh_count_true.add_obj(
                id, sham::DeviceBuffer<u32>(count, shamsys::instance::get_compute_scheduler_ptr()));

            sham::kernel_call(
                q,
                sham::MultiRef{
                    pcache, edges.xyz.get_spans().get(id), edges.hpart.get_spans().get(id)},
                sham::MultiRef{neigh_count_true.get(id), neigh_count_all.get(id)},
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
        });

        f64 max_true;
        f64 min_true;
        f64 mean_true;
        f64 stddev_true;
        f64 max_all;
        f64 min_all;
        f64 mean_all;
        f64 stddev_all;

        {
            std::vector<u32> all_max_true;
            neigh_count_true.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                all_max_true.push_back(*std::max_element(vec.begin(), vec.end()));
            });
            max_true = *std::max_element(all_max_true.begin(), all_max_true.end());
            max_true = shamalgs::collective::allreduce_max(max_true);
        }

        {
            std::vector<u32> all_max_all;
            neigh_count_all.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                all_max_all.push_back(*std::max_element(vec.begin(), vec.end()));
            });
            max_all = *std::max_element(all_max_all.begin(), all_max_all.end());
            max_all = shamalgs::collective::allreduce_max(max_all);
        }

        {
            std::vector<u32> all_min_true;
            neigh_count_true.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                all_min_true.push_back(*std::min_element(vec.begin(), vec.end()));
            });
            min_true = *std::min_element(all_min_true.begin(), all_min_true.end());
            min_true = shamalgs::collective::allreduce_min(min_true);
        }

        {
            std::vector<u32> all_min_all;
            neigh_count_all.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                all_min_all.push_back(*std::min_element(vec.begin(), vec.end()));
            });
            min_all = *std::min_element(all_min_all.begin(), all_min_all.end());
            min_all = shamalgs::collective::allreduce_min(min_all);
        }

        {
            std::vector<f64> sum_true_vec;
            std::vector<f64> count_true_vec;
            neigh_count_true.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                sum_true_vec.push_back(std::accumulate(vec.begin(), vec.end(), 0));
                count_true_vec.push_back(vec.size());
            });

            f64 sum_true   = std::accumulate(sum_true_vec.begin(), sum_true_vec.end(), 0.0);
            f64 count_true = std::accumulate(count_true_vec.begin(), count_true_vec.end(), 0.0);
            sum_true       = shamalgs::collective::allreduce_sum(sum_true);
            count_true     = shamalgs::collective::allreduce_sum(count_true);

            mean_true = sum_true / count_true;
        }

        {
            std::vector<f64> sum_all_vec;
            std::vector<f64> count_all_vec;
            neigh_count_all.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                sum_all_vec.push_back(std::accumulate(vec.begin(), vec.end(), 0.0));
                count_all_vec.push_back(vec.size());
            });

            f64 sum_all   = std::accumulate(sum_all_vec.begin(), sum_all_vec.end(), 0.0);
            f64 count_all = std::accumulate(count_all_vec.begin(), count_all_vec.end(), 0.0);
            sum_all       = shamalgs::collective::allreduce_sum(sum_all);
            count_all     = shamalgs::collective::allreduce_sum(count_all);

            mean_all = sum_all / count_all;
        }

        {
            std::vector<f64> sum_var_true_vec;
            std::vector<f64> count_true_vec;
            neigh_count_true.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                sum_var_true_vec.push_back(
                    std::accumulate(vec.begin(), vec.end(), 0.0, [mean_true](double acc, double x) {
                        return acc + (x - mean_true) * (x - mean_true);
                    }));
                count_true_vec.push_back(vec.size());
            });

            f64 sum_var_true
                = std::accumulate(sum_var_true_vec.begin(), sum_var_true_vec.end(), 0.0);
            f64 count_var_true = std::accumulate(count_true_vec.begin(), count_true_vec.end(), 0.0);
            sum_var_true       = shamalgs::collective::allreduce_sum(sum_var_true);
            count_var_true     = shamalgs::collective::allreduce_sum(count_var_true);

            stddev_true = std::sqrt(sum_var_true / count_var_true);
        }

        {
            std::vector<f64> sum_var_all_vec;
            std::vector<f64> count_all_vec;
            neigh_count_all.for_each([&](u64 id, sham::DeviceBuffer<u32> &buf) {
                std::vector<u32> vec = buf.copy_to_stdvec();
                sum_var_all_vec.push_back(
                    std::accumulate(vec.begin(), vec.end(), 0.0, [mean_all](double acc, double x) {
                        return acc + (x - mean_all) * (x - mean_all);
                    }));
                count_all_vec.push_back(vec.size());
            });

            f64 sum_var_all = std::accumulate(sum_var_all_vec.begin(), sum_var_all_vec.end(), 0.0);
            f64 count_var_all = std::accumulate(count_all_vec.begin(), count_all_vec.end(), 0.0);
            sum_var_all       = shamalgs::collective::allreduce_sum(sum_var_all);
            count_var_all     = shamalgs::collective::allreduce_sum(count_var_all);

            stddev_all = std::sqrt(sum_var_all / count_var_all);
        }

        if (!neigh_stats_json_loaded && !this->filename_dump.empty()) {
            load_neigh_stats_json(this->filename_dump);
            neigh_stats_json_loaded = true;
        }

        if (neigh_stats_json_loaded) {
            register_entry(
                edges.sim_time.data,
                max_true,
                min_true,
                mean_true,
                stddev_true,
                max_all,
                min_all,
                mean_all,
                stddev_all);
        }

        logger::raw_ln(
            "(true) max, min, mean, stddev =", max_true, min_true, mean_true, stddev_true);
        logger::raw_ln("(all) max, min, mean, stddev =", max_all, min_all, mean_all, stddev_all);
    }

    template<class Tvec>
    std::string ComputeNeighStats<Tvec>::_impl_get_tex() const {
        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::ComputeNeighStats<f64_3>;
