// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesFlagAccreteHard.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/sph/modules/SinkParticlesFlagAccreteHard.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec>
    void SinkParticlesFlagAccreteHard<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        auto &sink_positions = edges.sink_positions.data;
        auto &sink_radii     = edges.sink_accr_radii.data;

        if (sink_positions.size() != sink_radii.size()) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Sink positions and accretion radii must have the same size");
        }

        if (!sink_pos) {
            sink_pos = std::make_unique<sham::DeviceBuffer<Tvec>>(sink_positions.size(), dev_sched);
        }
        if (!sink_accr_radii) {
            sink_accr_radii
                = std::make_unique<sham::DeviceBuffer<Tscal>>(sink_radii.size(), dev_sched);
        }

        sink_pos->resize(sink_positions.size());
        sink_accr_radii->resize(sink_radii.size());

        sink_pos->copy_from_stdvec(sink_positions);
        sink_accr_radii->copy_from_stdvec(sink_radii);

        edges.positions.check_sizes(edges.part_counts.indexes);
        edges.sink_accretion_table.ensure_sizes(edges.part_counts.indexes);

        auto &pos_spans       = edges.positions.get_spans();
        auto &table_acc_spans = edges.sink_accretion_table.get_spans();

        u32 sink_count = shambase::narrow_or_throw<u32>(sink_positions.size());

        edges.part_counts.indexes.for_each([&](u64 id_patch, u32 part_count) {
            sham::kernel_call(
                q,
                sham::MultiRef{pos_spans.get(id_patch), *sink_pos, *sink_accr_radii},
                sham::MultiRef{table_acc_spans.get(id_patch)},
                part_count,
                [sink_count](
                    u32 id_a,
                    const Tvec *__restrict part_pos,
                    const Tvec *__restrict sink_pos,
                    const Tscal *__restrict sink_accr_radii,
                    u32 *__restrict sink_accretion_table) {
                    Tvec r_a = part_pos[id_a];

                    u32 result = u32_max;

                    for (u32 i_sink = 0; i_sink < sink_count; i_sink++) {
                        Tscal acc_radii = sink_accr_radii[i_sink];
                        Tvec d          = r_a - sink_pos[i_sink];

                        bool should_accrete = sycl::dot(d, d) <= acc_radii * acc_radii;
                        if (should_accrete) {
                            result = i_sink;
                            break;
                        }
                    }

                    sink_accretion_table[id_a] = result;
                });
        });
    }

    template<class Tvec>
    std::string SinkParticlesFlagAccreteHard<Tvec>::_impl_get_tex() const {
        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SinkParticlesFlagAccreteHard<f64_3>;
