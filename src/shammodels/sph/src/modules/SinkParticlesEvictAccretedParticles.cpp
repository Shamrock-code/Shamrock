// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesEvictAccretedParticles.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/sph/modules/SinkParticlesEvictAccretedParticles.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec>
    void SinkParticlesEvictAccretedParticles<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        sham::DeviceBuffer<u32> keep_flag(0, dev_sched);
        sham::DeviceBuffer<int> accr_flag(1, dev_sched);

        edges.part_counts.indexes.for_each([&](u64 id_patch, u32 Nobj) {
            auto &pdat      = edges.pdats.get(id_patch);
            auto &acc_table = edges.sink_accretion_table.get_spans().get(id_patch);

            keep_flag.resize(Nobj);
            accr_flag.fill(0);

            sham::kernel_call(
                q,
                sham::MultiRef{acc_table},
                sham::MultiRef{keep_flag, accr_flag},
                Nobj,
                [](u32 id_a,
                   const u32 *__restrict acc_table,
                   u32 *__restrict keep_flag,
                   int *__restrict accr_flag) {
                    bool keep       = acc_table[id_a] == u32_max;
                    keep_flag[id_a] = keep ? 1 : 0;

                    sycl::atomic_ref<
                        int,
                        sycl::memory_order_relaxed,
                        sycl::memory_scope_device,
                        sycl::access::address_space::global_space>
                        atomic_accr(accr_flag[0]);

                    if (!keep) {
                        atomic_accr.fetch_or(1);
                    }
                });

            int accr_flag_val = accr_flag.get_val_at_idx(0);

            if (accr_flag_val != 0) {

                sham::DeviceBuffer<u32> id_list_keep
                    = shamalgs::stream_compact(dev_sched, keep_flag, Nobj);

                pdat.keep_ids(
                    id_list_keep, shambase::narrow_or_throw<u32>(id_list_keep.get_size()));
            }
        });
    }

    template<class Tvec>
    std::string SinkParticlesEvictAccretedParticles<Tvec>::_impl_get_tex() const {
        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SinkParticlesEvictAccretedParticles<f64_3>;
